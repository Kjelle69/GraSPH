"""Read-only saved-state analysis and opt-in, sampled fluid force diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from grasph_collision import closest_point_triangle_batch


def nearest_solid(points, triangles, paths):
    """Exact unsigned distance to the supplied finite world-space triangles."""
    distance = np.full(len(points), np.inf)
    names = [None] * len(points)
    for triangle, path in zip(triangles, paths):
        closest = closest_point_triangle_batch(points, *triangle)
        delta = closest - points
        candidate = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        better = candidate < distance
        distance[better] = candidate[better]
        for index in np.flatnonzero(better):
            names[index] = str(path)
    return [float(d) if np.isfinite(d) else None for d in distance], names


def analyze_result(path, speed_threshold=1.0, top=5):
    """Inspect saved frames only. No integration or force reconstruction."""
    if not np.isfinite(speed_threshold) or speed_threshold < 0 or top < 1:
        raise ValueError("Use a finite nonnegative speed threshold and top >= 1")
    with np.load(path, allow_pickle=False) as data:
        times, positions, velocity = data["times"], data["positions"], data["velocity"]
        counts, ids = data["active_counts"], data["active_particle_ids"]
        mass = float(data["mass"])
        frames = []
        first_crossing = None
        for k, t in enumerate(times):
            n = int(counts[k])
            vv = np.asarray(velocity[k, :n], dtype=np.float64)
            speed = np.linalg.norm(vv, axis=1)
            fastest = np.argsort(-speed, kind="stable")[:top]
            fast = speed > speed_threshold
            particles = [{"id": int(ids[k, i]), "speed": float(speed[i]),
                          "position": positions[k, i].tolist()} for i in fastest]
            frame = {"time": float(t), "count": n, "over_threshold": int(fast.sum()),
                     "speed_p95": float(np.percentile(speed, 95)) if n else 0.,
                     "speed_rms": float(np.sqrt(np.mean(speed**2))) if n else 0.,
                     "kinetic_energy_j": float(.5 * mass * np.sum(speed**2)),
                     "fastest": particles}
            if first_crossing is None and np.any(fast):
                first_crossing = {"time": float(t), "previous_frame_time": float(times[k-1]) if k else None,
                                  "over_threshold": int(fast.sum()),
                                  "particles": [p for p in particles if p["speed"] > speed_threshold]}
            frames.append(frame)
        return {"result": str(Path(path).resolve()), "source": str(data["source_file"]),
                "note": "Saved-frame statistics, not per-step forces; threshold crossing is only bracketed.",
                "speed_threshold": speed_threshold, "first_crossing": first_crossing,
                "body_paths": data["body_paths"].tolist(), "frames": frames}


class ForceDiagnostics:
    """Sample only when requested. Persistent particle IDs survive sink removal.

    Caller supplies pre-step arrays and post-collision state before sink filtering.
    Recording must never change or replace the simulation arrays.
    """
    def __init__(self, path, start, end, interval, top=8, watch_ids=(), metadata=None):
        if not all(np.isfinite(x) for x in (start, end, interval)) or start < 0 or end < start or interval <= 0:
            raise ValueError("Diagnostics need 0 <= start <= end and a positive finite interval")
        if top < 1 or any(i < 0 for i in watch_ids):
            raise ValueError("Diagnostics need top >= 1 and nonnegative particle IDs")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start, self.end, self.interval = start, end, interval
        self.next_time, self.top = start, top
        self.watch_ids = tuple(watch_ids)
        with self.path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"type": "metadata", "schema": 1, **(metadata or {})}) + "\n")

    def due(self, time):
        if time + 1e-12 < self.next_time or time > self.end + 1e-12:
            return False
        self.next_time = self.start + (np.floor((time - self.start + 1e-12) / self.interval) + 1) * self.interval
        return True

    def record(self, *, time, dt, step, ids, before, after, acceleration,
               delta_velocity, density_rate, masks, triangles, triangle_paths, timings):
        speed_before = np.linalg.norm(before["velocity"], axis=1)
        speed_after = np.linalg.norm(after["velocity"], axis=1)
        chosen = np.unique(np.concatenate((
            np.argsort(-speed_before, kind="stable")[:self.top],
            np.argsort(-speed_after, kind="stable")[:self.top],
            np.flatnonzero(np.isin(ids, self.watch_ids)),
        )))
        distances, names = nearest_solid(before["position"][chosen], triangles, triangle_paths)
        particles = []
        for local, index in enumerate(chosen):
            particles.append({
                "id": int(ids[index]),
                "before": {key: np.asarray(value[index]).tolist() for key, value in before.items()},
                "after": {key: np.asarray(value[index]).tolist() for key, value in after.items()},
                "acceleration_m_s2": {key: np.asarray(value[index]).tolist() for key, value in acceleration.items()},
                "delta_velocity_m_s": {key: np.asarray(value[index]).tolist() for key, value in delta_velocity.items()},
                "density_rate_kg_m3_s": {key: float(value[index]) for key, value in density_rate.items()},
                "masks_before": {key: bool(value[index]) for key, value in masks.items()},
                "nearest_solid_before": names[local], "solid_distance_before_m": distances[local],
            })
        record = {"type": "sample", "time_before": float(time), "time_after": float(time + dt),
                  "dt": float(dt), "step": int(step), "count": len(ids), "phase_seconds": timings,
                  "speed_max_before": float(speed_before.max()) if len(ids) else 0.,
                  "speed_max_after": float(speed_after.max()) if len(ids) else 0.,
                  "speed_rms_after": float(np.sqrt(np.mean(speed_after**2))) if len(ids) else 0.,
                  "particles": particles}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze saved fluid snapshots without rerunning the solver")
    parser.add_argument("result")
    parser.add_argument("--out", required=True, help="Separate JSON analysis report")
    parser.add_argument("--speed-threshold", type=float, default=1.0)
    args = parser.parse_args()
    result, output = Path(args.result).resolve(), Path(args.out).resolve()
    if result == output or output.suffix.lower() != ".json":
        parser.error("Use a separate .json report path, not the input result")
    report = analyze_result(result, speed_threshold=args.speed_threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(output), "first_crossing": report["first_crossing"]}, indent=2))


if __name__ == "__main__":
    main()
