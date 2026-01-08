import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Protenix sweep across GPUs.")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU indices, e.g. '0,1,2,3'",
    )
    parser.add_argument(
        "--command",
        default="python3 scripts/protenix_unit_sweep.py",
        help="Command to run per shard.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    num_shards = len(gpus)
    procs = []

    for shard_id, gpu in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["SHARD_ID"] = str(shard_id)
        env["NUM_SHARDS"] = str(num_shards)
        print(f"Launching shard {shard_id}/{num_shards} on GPU {gpu}: {args.command}")
        procs.append(subprocess.Popen(args.command, shell=True, env=env))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"One or more shards failed: {exit_codes}")


if __name__ == "__main__":
    main()
