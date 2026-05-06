#!/usr/bin/env python3
"""
Convert /livox/lidar (livox_ros_driver2/CustomMsg) from ROS bag to PCD files.
Output: <outbase>/<bag_name>/view_000.pcd, view_001.pcd, ...
"""
import os, sys, argparse, struct
import numpy as np
import open3d as o3d
from rosbags.rosbag1 import Reader


def parse_custommsg(data):
    """Parse livox_ros_driver2/CustomMsg binary -> Nx3 float32 array."""
    off = 0
    # Header: seq(4) + stamp(8) + frame_id string
    off += 4 + 8
    fid_len = struct.unpack_from('<I', data, off)[0]; off += 4 + fid_len
    # timebase(8) + point_num(4) + lidar_id(1) + rsvd(3)
    off += 8 + 4 + 1 + 3
    # points array: uint32 count + N * 19 bytes
    n = struct.unpack_from('<I', data, off)[0]; off += 4
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    # CustomPoint: offset_time(4) + x(4) + y(4) + z(4) + reflectivity(1) + tag(1) + line(1) = 19
    raw = np.frombuffer(data, dtype=np.uint8, count=n * 19, offset=off)
    raw = raw.reshape(n, 19)
    xyz = np.frombuffer(raw[:, 4:16].tobytes(), dtype=np.float32).reshape(n, 3)
    mask = np.isfinite(xyz).all(axis=1) & (xyz != 0).any(axis=1)
    return xyz[mask]


def extract(bag_path, out_dir, interval=0.1, accumulate=1, max_frames=8,
            start_offset=0.0, duration=None):
    os.makedirs(out_dir, exist_ok=True)
    frame_idx, last_t, buffer = 0, None, []
    t0 = None

    with Reader(bag_path) as bag:
        conns = [c for c in bag.connections if c.topic == '/livox/lidar']
        if not conns:
            print(f"  WARNING: /livox/lidar not found"); return 0

        for conn, ts_ns, rawdata in bag.messages(connections=conns):
            if frame_idx >= max_frames:
                break
            ts = ts_ns * 1e-9
            if t0 is None:
                t0 = ts
            rel = ts - t0
            if rel < start_offset:
                continue
            if duration is not None and rel > start_offset + duration:
                break
            if last_t is not None and (ts - last_t) < interval:
                continue
            pts = parse_custommsg(bytes(rawdata))
            if len(pts) < 50:
                continue
            buffer.append(pts)
            last_t = ts
            if len(buffer) < accumulate:
                continue
            merged = np.vstack(buffer); buffer = []
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
            path = os.path.join(out_dir, f"view_{frame_idx:03d}.pcd")
            o3d.io.write_point_cloud(path, pcd)
            print(f"  {path}  ({len(merged)} pts)")
            frame_idx += 1

    print(f"Done: {frame_idx} frames -> {out_dir}")
    return frame_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="path to a .bag file or directory of .bag files")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--outdir", default="./output")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--start", type=float, default=0.0,
                        help="start offset in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="time window in seconds (short window = high overlap)")
    args = parser.parse_args()

    if os.path.isfile(args.bag):
        bags = [(args.bag, os.path.basename(args.bag))]
    else:
        bags = [(os.path.join(args.bag, f), f)
                for f in sorted(os.listdir(args.bag)) if f.endswith(".bag")]
    if not bags:
        sys.exit(f"No .bag files found")
    for bag_path, bag_file in bags:
        print(f"\n[{bag_file}] -> {args.outdir}")
        extract(bag_path, args.outdir, args.interval, args.accumulate,
                args.frames, args.start, args.duration)


if __name__ == "__main__":
    main()
