"""Whole-shape rigid contact regressions, independent of fluid resolution."""
from types import SimpleNamespace
import unittest

import numpy as np

from grasph_rigid import (
    _convex_pair_contact, _contact_samples, _oriented_surface_query,
    axis_angle_matrix, resolve_body_contacts,
)


def cube(path, center=(0, 0, 0), half=(1, 1, 1), mass=1., joint='free'):
    center, half = np.asarray(center, dtype=float), np.asarray(half, dtype=float)
    vertices = center + np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                  [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])*half
    faces = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
    triangles = np.array([vertices[list(t)] for a,b,c,d in faces for t in [(a,b,c),(a,c,d)]])
    samples = np.unique(np.vstack((vertices, triangles.mean(axis=1),
        .5*(triangles[:,0]+triangles[:,1]), .5*(triangles[:,1]+triangles[:,2]),
        .5*(triangles[:,2]+triangles[:,0]))), axis=0)
    body = SimpleNamespace(path=path, center=center, mass=mass, joint_type=joint,
        joint_axis=np.array([1.,0,0]), collision_shape='convex', collision_margin=.002,
        collision_samples=samples, inertia=mass/3*(np.sum(half**2)-half**2))
    collider = SimpleNamespace(name=path, triangles=triangles, voxel_size=.025)
    state = dict(translation=np.zeros(3), rotation=np.eye(3), velocity=np.zeros(3), omega=np.zeros(3))
    return body, collider, state


def solve(a, b, iterations=4):
    bodies={a[0].path:a[0],b[0].path:b[0]}
    states={a[0].path:a[2],b[0].path:b[2]}
    scene=SimpleNamespace(dynamic_bodies=bodies, collider_materials={p:{'friction':0.,'restitution':0.} for p in bodies}, hinges={})
    # Deliberately stale snapshots: the solver must use corrected live poses.
    shifts={p:s['translation'].copy() for p,s in states.items()}
    rotations={p:s['rotation'].copy() for p,s in states.items()}
    return resolve_body_contacts(scene,[a[1],b[1]],states,shifts,{},rotations,
                                 {p:x.center for p,x in bodies.items()},{},iterations=iterations)


def contact(a,b):
    return _convex_pair_contact(a[0],a[2],a[1],b[0],b[2],b[1])


def independent_box_depth(a,b):
    """Independent 15-axis OBB overlap check (no collision margin)."""
    center_a=a[0].center+a[2]['translation']; center_b=b[0].center+b[2]['translation']
    ha=np.ptp(a[1].triangles.reshape(-1,3),axis=0)/2
    hb=np.ptp(b[1].triangles.reshape(-1,3),axis=0)/2
    ra,rb=a[2]['rotation'],b[2]['rotation']
    axes=[*ra.T,*rb.T]+[np.cross(x,y) for x in ra.T for y in rb.T]
    depths=[]
    for axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-10:continue
        axis=axis/length
        depths.append(ha@np.abs(ra.T@axis)+hb@np.abs(rb.T@axis)-abs((center_a-center_b)@axis))
    return min(depths)


class ConvexBodyContactTests(unittest.TestCase):
    def test_small_cube_hits_between_large_cube_surface_samples(self):
        big=cube('/A'); small=cube('/Z',center=(1.03,.43,.31),half=(.15,)*3)
        points,_=_contact_samples(big[0],big[2])
        distance,_=_oriented_surface_query(points,small[1].triangles)
        self.assertGreater(distance.min(),.0125)  # Old one-way test misses it.
        self.assertGreater(independent_box_depth(big,small),.1)
        contacts,_=solve(big,small)
        self.assertGreater(contacts,0)
        self.assertLess(independent_box_depth(big,small),1e-9)
        self.assertIsNone(contact(big,small))

    def test_swapped_names_and_pair_order_give_same_result(self):
        outcomes=[]
        for names in [('/A','/Z'),('/Z','/A')]:
            big=cube(names[0]); small=cube(names[1],center=(1.03,.43,.31),half=(.15,)*3)
            small[2]['velocity'][:]=[-1.,0,0]
            solve(big,small)
            outcomes.append([s[2][key].copy() for s in [big,small] for key in ['translation','velocity','omega']])
        np.testing.assert_allclose(outcomes[0],outcomes[1],atol=1e-10)

    def test_mass_weighted_separation_and_momentum(self):
        a=cube('/A',half=(.5,)*3,mass=1.)
        b=cube('/B',center=(.9,0,0),half=(.5,)*3,mass=3.)
        a[2]['velocity'][0]=1.
        initial_momentum=a[2]['velocity']+3*b[2]['velocity']
        solve(a,b)
        np.testing.assert_allclose(a[2]['translation']+3*b[2]['translation'],0,atol=1e-12)
        np.testing.assert_allclose(a[2]['velocity']+3*b[2]['velocity'],initial_momentum,atol=1e-12)
        np.testing.assert_allclose(a[2]['omega'],0,atol=1e-12)
        np.testing.assert_allclose(b[2]['omega'],0,atol=1e-12)
        self.assertAlmostEqual(a[2]['velocity'][0],.25)
        self.assertAlmostEqual(b[2]['velocity'][0],.25)

    def test_fixed_body_is_not_moved_or_accelerated(self):
        fixed=cube('/Z',joint='fixed'); moving=cube('/A',center=(1.03,.43,.31),half=(.15,)*3)
        moving[2]['velocity'][0]=-1.
        solve(fixed,moving)
        for key in ['translation','velocity','omega']:
            np.testing.assert_array_equal(fixed[2][key],0)
        self.assertLess(independent_box_depth(fixed,moving),0)

    def test_prismatic_correction_respects_allowed_axis(self):
        sliding=cube('/A',joint='prismatic'); other=cube('/B',center=(0,1.8,0))
        other[2]['velocity'][1]=-1.
        solve(sliding,other)
        np.testing.assert_array_equal(sliding[2]['translation'],0)
        np.testing.assert_array_equal(sliding[2]['omega'],0)
        np.testing.assert_array_equal(sliding[2]['velocity'],0)
        self.assertLess(independent_box_depth(sliding,other),0)

    def test_containment_uses_full_exit_distance(self):
        big=cube('/A',joint='fixed'); small=cube('/B',center=(.2,.1,0),half=(.1,)*3)
        self.assertGreater(contact(big,small)[1],.8)
        solve(big,small)
        self.assertLess(independent_box_depth(big,small),0)

    def test_rotated_boxes_match_independent_sat(self):
        rng=np.random.default_rng(20261016)
        for _ in range(120):
            a=cube('/A',half=(.4,.2,.15)); b=cube('/B',center=rng.uniform(-.8,.8,3),half=(.3,.15,.2))
            a[0].collision_margin=b[0].collision_margin=0.
            for shape in [a,b]:
                shape[2]['rotation']=axis_angle_matrix(rng.normal(size=3),rng.uniform(-3,3))
            depth=independent_box_depth(a,b)
            hit=contact(a,b)
            self.assertEqual(hit is not None,depth>1e-10)
            if hit is not None:
                self.assertAlmostEqual(hit[1],depth,places=9)
                solve(a,b)
                self.assertLess(independent_box_depth(a,b),1e-9)

    def test_crossed_thin_edges_detect_without_vertex_containment(self):
        a=cube('/A',half=(1.,.06,.06))
        b=cube('/B',center=(0,0,.08),half=(.06,1.,.06))
        self.assertIsNotNone(contact(a,b))
        solve(a,b)
        self.assertLess(independent_box_depth(a,b),0)

    def test_repeated_head_on_contacts_do_not_gain_kinetic_energy(self):
        a=cube('/A',center=(-.49,0,0),half=(.5,)*3)
        b=cube('/B',center=(.49,0,0),half=(.5,)*3)
        a[2]['velocity'][0]=1.; b[2]['velocity'][0]=-1.
        for _ in range(10):
            before=sum(.5*s[0].mass*np.dot(s[2]['velocity'],s[2]['velocity'])
                       +.5*np.dot(s[0].inertia*s[2]['omega'],s[2]['omega']) for s in [a,b])
            solve(a,b)
            after=sum(.5*s[0].mass*np.dot(s[2]['velocity'],s[2]['velocity'])
                      +.5*np.dot(s[0].inertia*s[2]['omega'],s[2]['omega']) for s in [a,b])
            self.assertLessEqual(after,before+1e-12)
        self.assertLess(independent_box_depth(a,b),0)

    def test_multiple_contacts_converge_with_fresh_poses(self):
        shapes=[cube('/A',center=(0,0,0),half=(.5,)*3,joint='fixed'),
                cube('/B',center=(.9,0,0),half=(.5,)*3),
                cube('/C',center=(1.8,0,0),half=(.5,)*3)]
        bodies={b.path:b for b,c,s in shapes}; states={b.path:s for b,c,s in shapes}
        scene=SimpleNamespace(dynamic_bodies=bodies,collider_materials={},hinges={})
        resolve_body_contacts(scene,[c for b,c,s in shapes],states,{}, {}, {},
                              {b.path:b.center for b,c,s in shapes},{},iterations=2)
        for a,b in zip(shapes,shapes[1:]):
            self.assertLess(independent_box_depth(a,b),1e-7)
        np.testing.assert_array_equal(shapes[0][2]['translation'],0)


if __name__=='__main__':
    unittest.main()
