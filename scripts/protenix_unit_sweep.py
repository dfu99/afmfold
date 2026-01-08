import argparse
import itertools
import os

import mdtraj as md
import numpy as np

import afmfold.domain as domain
from afmfold.domain import compute_domain_distance
from afmfold.runner.batch_inference import inference_jsons


def parse_list(value, cast=float):
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items]


def target_from_pdb(pdb_path, domain_pairs):
    traj = md.load(pdb_path)
    dists = []
    for d1, d2 in domain_pairs:
        d = compute_domain_distance(traj, d1, d2)
        dists.append(d.item() * 10.0)  # nm -> Å
    return np.array(dists)


def run_target(json_file, out_dir, domain_pairs, target, seed, y_max, t_start, n_sample):
    kwargs = {
        "sample_diffusion.N_sample": n_sample,
        "guidance_kwargs.t_start": t_start,
        "guidance_kwargs.manual": target,
        "guidance_kwargs.scaling_kwargs.func_type": "constant",
        "guidance_kwargs.scaling_kwargs.y_max": y_max,
        "guidance_kwargs.domain_pairs": domain_pairs,
    }
    inference_jsons(json_file, out_dir, seeds=(seed,), **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Protenix-only EO/EC sweep with sharding.")
    parser.add_argument("--json-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bc-pdb", required=True)
    parser.add_argument("--name", default="4G1E")
    parser.add_argument("--eo-offsets", default="20,30,40")
    parser.add_argument("--ec-offsets", default="-5,-10")
    parser.add_argument("--y-max-list", default="0.6,1.0,2.0")
    parser.add_argument("--t-start-list", default="0.1,0.2,0.4")
    parser.add_argument("--seeds", default="101,102,103,104")
    parser.add_argument("--n-sample", type=int, default=2)
    args = parser.parse_args()

    domain_pairs = [
        (domain.INTEGRIN_DOMAINS["alpha_head_thigh"], domain.INTEGRIN_DOMAINS["alpha_calf"]),
        (domain.INTEGRIN_DOMAINS["alpha_calf"], domain.INTEGRIN_DOMAINS["alpha_coil"]),
        (domain.INTEGRIN_DOMAINS["beta_head_hybrid_egf1"], domain.INTEGRIN_DOMAINS["beta_tail_egf2_3_4_btail"]),
        (domain.INTEGRIN_DOMAINS["beta_tail_egf2_3_4_btail"], domain.INTEGRIN_DOMAINS["beta_coil"]),
    ]

    bc_dist = target_from_pdb(args.bc_pdb, domain_pairs)

    eo_offsets = parse_list(args.eo_offsets, float)
    ec_offsets = parse_list(args.ec_offsets, float)
    y_max_list = parse_list(args.y_max_list, float)
    t_start_list = parse_list(args.t_start_list, float)
    seeds = parse_list(args.seeds, int)

    jobs = []
    for eo_offset, y_max, t_start, seed in itertools.product(
        eo_offsets, y_max_list, t_start_list, seeds
    ):
        jobs.append(("EO", bc_dist + eo_offset, seed, y_max, t_start))
    for ec_offset, y_max, t_start, seed in itertools.product(
        ec_offsets, y_max_list, t_start_list, seeds
    ):
        jobs.append(("EC", bc_dist + ec_offset, seed, y_max, t_start))

    shard_id = int(os.environ.get("SHARD_ID", "0"))
    num_shards = int(os.environ.get("NUM_SHARDS", "1"))

    os.makedirs(args.out_dir, exist_ok=True)

    for idx, (label, target, seed, y_max, t_start) in enumerate(jobs):
        if idx % num_shards != shard_id:
            continue
        print(
            f"[{idx + 1}/{len(jobs)}][{label}] seed={seed} y_max={y_max} "
            f"t_start={t_start} target={target}"
        )
        run_target(
            args.json_file,
            args.out_dir,
            domain_pairs,
            target,
            seed,
            y_max,
            t_start,
            args.n_sample,
        )


if __name__ == "__main__":
    main()
