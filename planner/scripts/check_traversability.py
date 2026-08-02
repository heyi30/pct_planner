#!/usr/bin/env python3
"""Check traversability between all waypoint pairs in a JSON file.

Reads a JSON list of waypoint entries (each with a "pose" [x, y, z]),
plans a path between every pair of waypoints (i -> j, i < j), and
writes the pairs that cannot be traversed to <name>_unreachable.json
for connectivity inspection.

Optionally saves all planned trajectories and their corresponding
point pairs to <name>_traj.json when --save-traj is given.

Usage:
    python3 check_traversability.py /path/to/waypoints.json
    python3 check_traversability.py /path/to/waypoints.json --map scene_map_sparse --save-traj
"""

import argparse
import json
import os

from Planner import Planner


def load_waypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of waypoint entries")
    for i, entry in enumerate(data):
        pose = entry.get('pose')
        if not (isinstance(pose, list) and len(pose) == 3):
            raise ValueError(
                f"entry {i} ({entry.get('name')!r}) has no valid 'pose' [x, y, z]"
            )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_file', type=str,
                        help="input JSON file with waypoint entries")
    parser.add_argument('--map', type=str, default='chagee_planner',
                        help="sparse tomogram pickle name under rsc/tomogram/ (without .pickle)")
    parser.add_argument('--save-traj', action='store_true',
                        help="save all planned trajectories and point pairs to <name>_traj.json")
    args = parser.parse_args()

    entries = load_waypoints(args.json_file)
    if len(entries) < 2:
        raise ValueError("need at least two waypoints to check")

    n_pairs = len(entries) * (len(entries) - 1) // 2
    print(f"Loaded {len(entries)} waypoints from {args.json_file}")
    print(f"Checking {n_pairs} waypoint pairs")
    print(f"Loading map: {args.map}.pickle")
    planner = Planner(map_name=args.map)

    unreachable = []
    trajectories = []
    pair_idx = 0
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            pair_idx += 1
            start = entries[i]['pose']
            goal = entries[j]['pose']
            print(f"\n[{pair_idx}/{n_pairs}] {entries[i].get('name')} -> {entries[j].get('name')}")
            print(f"    start = {start}, goal = {goal}")
            path = planner.plan(start, goal)
            if path is None:
                print("    UNREACHABLE")
                unreachable.append({
                    'start_name': entries[i].get('name'),
                    'start_pose': start,
                    'goal_name': entries[j].get('name'),
                    'goal_pose': goal,
                })
            else:
                print(f"    reachable, waypoints: {path.shape[0]}")
                if args.save_traj:
                    trajectories.append({
                        'start_name': entries[i].get('name'),
                        'start_pose': start,
                        'goal_name': entries[j].get('name'),
                        'goal_pose': goal,
                        'trajectory': path.tolist(),
                    })

    out_path = os.path.splitext(args.json_file)[0] + '_unreachable.json'
    with open(out_path, 'w') as f:
        json.dump(unreachable, f, indent=2, ensure_ascii=False)

    print(f"\n{len(unreachable)} / {n_pairs} pairs unreachable")
    print(f"Saved unreachable pairs to: {out_path}")

    if args.save_traj:
        traj_path = os.path.splitext(args.json_file)[0] + '_traj.json'
        with open(traj_path, 'w') as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(trajectories)} trajectories to: {traj_path}")

    return 1 if unreachable else 0


if __name__ == '__main__':
    raise SystemExit(main())
