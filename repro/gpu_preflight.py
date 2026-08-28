#!/usr/bin/env python3
"""Read nvidia-smi before using one physical GPU; never initialize CUDA.

This is a point-in-time occupancy check, not a scheduler reservation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys


def query_rows(query: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=15,
    )
    return [[value.strip() for value in row] for row in
            csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row]


def inspect_gpu(index: int, min_free_mib: int = 23000,
                max_used_mib: int = 256, max_util_percent: int = 5) -> dict:
    gpus = query_rows(
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    )
    matching = [row for row in gpus if len(row) == 7 and int(row[0]) == index]
    if len(matching) != 1:
        raise RuntimeError(f"Physical GPU {index} was not found uniquely")
    gpu = matching[0]
    apps = query_rows("--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
    occupants = [row for row in apps if row[0] == gpu[1]]
    report = {"index": int(gpu[0]), "uuid": gpu[1], "name": gpu[2],
              "total_mib": int(gpu[3]), "used_mib": int(gpu[4]),
              "free_mib": int(gpu[5]), "util_percent": int(gpu[6]),
              "compute_processes": occupants}
    if (occupants or report["used_mib"] > max_used_mib
            or report["util_percent"] > max_util_percent):
        report.update(status="occupied", reason="Existing GPU activity; refusing to start")
    elif report["free_mib"] < min_free_mib:
        report.update(status="insufficient_memory",
                      reason=f"Need at least {min_free_mib} MiB free for this configured trial")
    else:
        report["status"] = "idle"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=23000)
    parser.add_argument("--max-used-mib", type=int, default=256)
    parser.add_argument("--max-util-percent", type=int, default=5)
    parser.add_argument("--uuid-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.gpu_index, args.min_free_mib, args.max_used_mib, args.max_util_percent) < 0:
        parser.error("GPU index and thresholds must be nonnegative")
    if args.max_util_percent > 100:
        parser.error("--max-util-percent cannot exceed 100")
    try:
        report = inspect_gpu(args.gpu_index, args.min_free_mib,
                             args.max_used_mib, args.max_util_percent)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    if report["status"] != "idle":
        print(json.dumps(report), file=sys.stderr)
        return 3
    print(report["uuid"] if args.uuid_only else json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
