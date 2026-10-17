"""Shared fixed joints: explicit body/body or world/body constraints.

Experimental impulse-level bilateral constraints. Break limits are engineering
thresholds (N/Nm), not a material fracture law. Impact loads depend on timestep.
No mesh fragmentation or automatic topology is performed here.
"""
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FixedJoint:
    path: str
    body_a: str | None  # None denotes world
    body_b: str
    anchor: np.ndarray  # world-space reference anchor, metres
    break_force: float = np.inf
    break_torque: float = np.inf

    def __post_init__(self):
        if not self.path or not self.body_b or self.body_a == self.body_b:
            raise ValueError('Joint requires a name and distinct endpoints')
        anchor = np.asarray(self.anchor, dtype=float)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            raise ValueError('Joint anchor must be a finite 3D position')
        object.__setattr__(self, 'anchor', anchor.copy())
        for limit in (self.break_force, self.break_torque):
            if np.isnan(limit) or limit <= 0:
                raise ValueError('Joint limits must be positive or infinity')


def load_fixed_joints(stage, bodies):
    """Read explicit GraSPH joints or ordinary USD fixed joints, without duplication."""
    from pxr import UsdPhysics, UsdGeom

    def body_path(target):
        matches = [p for p, b in bodies.items()
                   if str(target) in (p, b.driver_path)]
        if len(matches) != 1:
            raise ValueError(f'Joint endpoint {target} must identify exactly one Dynamic Body')
        return matches[0]

    joints = []
    for prim in stage.Traverse():
        rel = prim.GetRelationship('grasph:bodyB')
        if rel:
            kind = str(prim.GetAttribute('grasph:jointType').Get() or '')
            if kind != 'fixed':
                continue  # legacy world joints retain their original solver
            a = prim.GetRelationship('grasph:bodyA').GetTargets()
            b = rel.GetTargets()
            if len(a) > 1 or len(b) != 1:
                raise ValueError(f'{prim.GetPath()}: invalid joint endpoints')
            anchor = prim.GetAttribute('grasph:jointAnchor').Get()
            enabled = bool(prim.GetAttribute('grasph:breakable').Get())
            force = prim.GetAttribute('grasph:breakForce').Get() if enabled else np.inf
            torque = prim.GetAttribute('grasph:breakTorque').Get() if enabled else np.inf
        elif prim.IsA(UsdPhysics.FixedJoint):
            if prim.GetAttribute('grasph:managedJoint').Get():
                continue
            joint = UsdPhysics.FixedJoint(prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            swapped = not b and bool(a)
            if not b and a:
                a, b = [], a
            if len(a) > 1 or len(b) != 1:
                raise ValueError(f'{prim.GetPath()}: unsupported fixed joint endpoints')
            local = (joint.GetLocalPos0Attr().Get() if swapped else joint.GetLocalPos1Attr().Get())
            local = (0, 0, 0) if local is None else local
            xf = UsdGeom.XformCache(stage.GetStartTimeCode()).GetLocalToWorldTransform(stage.GetPrimAtPath(b[0]))
            from pxr import Gf
            anchor = xf.Transform(Gf.Vec3d(*local))
            # USD defaults unset break limits to an effectively infinite value.
            force = joint.GetBreakForceAttr().Get()
            torque = joint.GetBreakTorqueAttr().Get()
            force = np.inf if force is None else force
            torque = np.inf if torque is None else torque
            # Old exports also encode fixed-to-world constraints on the body.
            if not a and bodies[body_path(b[0])].joint_type == 'fixed':
                continue
        else:
            continue
        joints.append(FixedJoint(str(prim.GetPath()), body_path(a[0]) if a else None,
                                 body_path(b[0]), anchor, float(force), float(torque)))
    return tuple(joints)


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])


def _rotation_error(rotation):
    # Eigenvector extraction remains well-defined near pi, unlike sin(theta).
    angle = float(np.arccos(np.clip((np.trace(rotation)-1)*.5, -1, 1)))
    vector = np.array([rotation[2, 1]-rotation[1, 2],
                       rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    if angle < 1e-7:
        return .5*vector
    if abs(np.sin(angle)) > 1e-6:
        return vector * (angle/(2*np.sin(angle)))
    values, axes = np.linalg.eigh((rotation + rotation.T)*.5)
    return axes[:, np.argmax(values)] * angle


class JointSystem:
    """Run-local fixed-joint state shared by every rigid simulation path.

    All active constraints are solved together. Closed loops use the minimum
    norm reaction solution; this is not a unique material stress distribution.
    Contacts remain per-body; compound contact manifolds are future work.
    """
    def __init__(self, scene, states, event_path):
        self.scene, self.states = scene, states
        self.joints = tuple(getattr(scene, 'fixed_joints', ()))
        self.broken = set()
        self.events = []
        self.last_time = -np.inf
        self.event_path = Path(event_path)
        if not self.joints:
            return
        if len({j.path for j in self.joints}) != len(self.joints):
            raise ValueError('Joint paths must be unique')
        self.paths = sorted({p for j in self.joints for p in (j.body_a, j.body_b) if p is not None})
        for p in self.paths:
            if p not in states or scene.dynamic_bodies[p].joint_type != 'free':
                raise ValueError(f'{p}: fixed joints need a simulated free body; do not mix with legacy world constraints')
        self.index = {p:i for i,p in enumerate(self.paths)}
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_path.write_text('', encoding='utf-8')
        self._exclusions()
        print(f'Fixed joints: {len(self.joints)}; break events: {self.event_path}')

    def _components(self):
        adjacency = {p:set() for p in self.paths}
        world = set()
        for j in self.joints:
            if j.path in self.broken:
                continue
            if j.body_a is None:
                world.add(j.body_b)
            else:
                adjacency[j.body_a].add(j.body_b)
                adjacency[j.body_b].add(j.body_a)
        unseen = set(self.paths)
        result = []
        while unseen:
            pending, group = [min(unseen)], set()
            while pending:
                p = pending.pop()
                if p not in group:
                    group.add(p)
                    pending.extend(adjacency[p]-group)
            unseen -= group
            result.append((sorted(group), bool(group & world)))
        return result

    def _exclusions(self):
        self.scene.joint_collision_exclusions = {
            frozenset(pair) for component, _ in self._components()
            for pair in combinations(component, 2)
        }

    def _system(self, active, dt):
        size = 6*len(self.paths)
        inverse = np.zeros((size, size))
        velocity = np.zeros(size)
        for p, i in self.index.items():
            b, s = self.scene.dynamic_bodies[p], self.states[p]
            start = 6*i
            inverse[start:start+3, start:start+3] = np.eye(3)/b.mass
            inverse[start+3:start+6, start+3:start+6] = s['rotation'] @ np.diag(1/np.asarray(b.inertia)) @ s['rotation'].T
            velocity[start:start+6] = np.r_[s['velocity'], s['omega']]
        jacobian = np.zeros((6*len(active), size))
        error = np.zeros(6*len(active))
        for i, j in enumerate(active):
            points, rotations = [], []
            for p, sign in ((j.body_a, -1), (j.body_b, 1)):
                if p is None:
                    points.append(j.anchor)
                    rotations.append(np.eye(3))
                    continue
                b, s = self.scene.dynamic_bodies[p], self.states[p]
                r = s['rotation'] @ (j.anchor-b.center)
                points.append(b.center+s['translation']+r)
                rotations.append(s['rotation'])
                col, row = 6*self.index[p], 6*i
                jacobian[row:row+3, col:col+3] = sign*np.eye(3)
                jacobian[row:row+3, col+3:col+6] = -sign*_skew(r)
                jacobian[row+3:row+6, col+3:col+6] = sign*np.eye(3)
            error[6*i:6*i+3] = points[1]-points[0]
            error[6*i+3:6*i+6] = _rotation_error(rotations[1] @ rotations[0].T)
        bias = .2*error/dt
        effective = jacobian @ inverse @ jacobian.T
        impulses = np.linalg.lstsq(effective, -(jacobian @ velocity+bias), rcond=1e-12)[0]
        return velocity, inverse @ jacobian.T @ impulses, impulses.reshape(-1, 6)

    def solve(self, dt, time):
        if not self.joints:
            return
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(time) or time <= self.last_time:
            raise ValueError('Joint timestep must be positive and time must advance')
        # Redistribute reactions after each set of breaks, always using the same
        # pre-constraint velocity. No rejected/trial impulse is applied.
        for _ in range(len(self.joints)+1):
            active = [j for j in self.joints if j.path not in self.broken]
            if not active:
                break
            velocity, delta, impulses = self._system(active, dt)
            new_events = []
            for j, impulse in zip(active, impulses):
                force, torque = np.linalg.norm(impulse[:3])/dt, np.linalg.norm(impulse[3:])/dt
                if force > j.break_force or torque > j.break_torque:
                    new_events.append(dict(joint=j.path, time=float(time), force_N=float(force), torque_Nm=float(torque)))
            if new_events:
                for event in new_events:
                    self.broken.add(event['joint'])
                    self.events.append(event)
                    with self.event_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(event)+'\n')
                    print(f'JOINT BREAK t={time:.6f}s {event["joint"]}: {event["force_N"]:.3f} N, {event["torque_Nm"]:.3f} Nm')
                continue
            result = velocity+delta
            for p, i in self.index.items():
                self.states[p]['velocity'][:] = result[6*i:6*i+3]
                self.states[p]['omega'][:] = result[6*i+3:6*i+6]
            break
        self.last_time = time
        self._exclusions()

    def project_poses(self):
        if not self.joints:
            return
        for paths, pinned in self._components():
            if len(paths) == 1 and not pinned:
                continue
            bodies = [self.scene.dynamic_bodies[p] for p in paths]
            states = [self.states[p] for p in paths]
            masses = [b.mass for b in bodies]
            reference = np.array([b.center for b in bodies])
            ref_com = np.average(reference, axis=0, weights=masses)
            rotation = np.eye(3) if pinned else states[0]['rotation'].copy()
            com = ref_com if pinned else np.average(
                [b.center+s['translation'] for b,s in zip(bodies, states)], axis=0, weights=masses)
            offsets = (reference-ref_com) @ rotation.T
            for b, s, r in zip(bodies, states, offsets):
                s['translation'][:] = com+r-b.center
                s['rotation'][:] = rotation
