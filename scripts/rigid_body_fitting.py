import os
import sys
#sys.modules["deepspeed"] = None
import numpy as np
import argparse
import glob
import math

from afmfold.rigid_body_fitting import run_rigid_body_fitting, save_args_to_file

ALL_ARGS = [
    "all_pred_cc", "all_pred_rots", "all_pred_images", "all_pred_coords",
    "top_cc", "top_rots", "top_pred_images", "top_pred_coords",
    "ref_images", "ref_domain_distance", "pred_domain_distance",
]
TOP_ARGS = [
    "top_cc", "top_rots", "top_pred_images", "top_pred_coords",
    "ref_images", "ref_domain_distance", "pred_domain_distance",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--ref-pdb", type=str, default="storage/3a5i.pdb")
    parser.add_argument("--resolution-nm", type=float, default=0.98)
    parser.add_argument("--prove-radius-mean", type=float, default=2.0)
    parser.add_argument("--prove-radius-range", type=float, default=0.0)
    parser.add_argument("--prove-radius-step", type=float, default=1.0)
    parser.add_argument("--min-z", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--stop-at", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=5)
    parser.add_argument("--name", type=str, default="fitting")
    parser.add_argument("--use-ref-structure", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--skip-finished", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.prove_radius_range == 0.0:
        prove_radius_range = None
        prove_radius = args.prove_radius_mean
    else:
        prove_radius_range = (args.prove_radius_mean - args.prove_radius_range, args.prove_radius_mean + args.prove_radius_range)
        prove_radius = None
        
    if len(args.output_dirs) == 1:
        if args.skip_finished:
            if args.use_ref_structure:
                assert os.path.exists(os.path.join(args.output_dirs, "predictions", "inputs.npz"))
                assert not os.path.exists(os.path.join(args.output_dirs, "predictions", f"{args.name}_ref.npz"))
                
            else:
                assert os.path.exists(os.path.join(args.output_dirs, "predictions", "inputs.npz"))
                assert not os.path.exists(os.path.join(args.output_dirs, "predictions", f"{args.name}.npz"))
                
        else:
            assert os.path.exists(os.path.join(args.output_dirs, "predictions", "inputs.npz"))
            
        total_summary = run_rigid_body_fitting(
            [args.output_dirs,], 
            ref_pdb=args.ref_pdb, 
            resolution_nm=args.resolution_nm, 
            prove_radius_range=prove_radius_range, 
            prove_radius_step=args.prove_radius_step,
            prove_radius=prove_radius,
            min_z=args.min_z, 
            steps=args.steps, 
            stop_at=args.stop_at, 
            batchsize=args.batchsize, 
            use_ref_structure=args.use_ref_structure, 
            device=args.device, 
            )
        
        if args.save_all:
            total_summary = {k: v for k, v in total_summary.items() if k in ALL_ARGS}
        else:
            total_summary = {k: v for k, v in total_summary.items() if k in TOP_ARGS}
        
        if args.use_ref_structure:
            suffix = "_ref"
        else:
            suffix = ""
            
        if len(total_summary) > 0:
            np.savez_compressed(os.path.join(args.output_dirs, "predictions", f"{args.name}{suffix}.npz"), **total_summary)
            save_args_to_file(args, os.path.join(args.output_dirs, "predictions", f"{args.name}{suffix}.json"))
            
    else:
        assert all(os.listdir(d) == ["predictions"] for d in args.output_dirs)
        
        if args.use_ref_structure:
            suffix = "_ref"
        else:
            suffix = ""
        
        if args.skip_finished:
            if args.use_ref_structure:
                valid_output_dirs = [
                    output_dir for output_dir in args.output_dirs
                    if len(glob.glob(os.path.join(output_dir, "predictions", "inputs.npz"))) > 0
                    and len(glob.glob(os.path.join(output_dir, "predictions", f"{args.name}_ref.npz"))) == 0
                    ]
            else:
                valid_output_dirs = [
                    output_dir for output_dir in args.output_dirs
                    if len(glob.glob(os.path.join(output_dir, "predictions", "inputs.npz"))) > 0
                    and len(glob.glob(os.path.join(output_dir, "predictions", f"{args.name}.npz"))) == 0
                    ]
        else:
            valid_output_dirs = [
                output_dir for output_dir in args.output_dirs
                if len(glob.glob(os.path.join(output_dir, "predictions", "inputs.npz"))) > 0
                ]
            
        if len(valid_output_dirs) < args.batchsize:
            batchs = [valid_output_dirs]
        else:
            batchs = [valid_output_dirs[i*args.batchsize:(i+1)*args.batchsize] for i in range(math.ceil(len(valid_output_dirs) / args.batchsize))]
        
        for batch in batchs:
            total_summary = run_rigid_body_fitting(
                batch, 
                ref_pdb=args.ref_pdb, 
                resolution_nm=args.resolution_nm, 
                prove_radius_range=prove_radius_range, 
                prove_radius_step=args.prove_radius_step,
                prove_radius=prove_radius,
                min_z=args.min_z, 
                steps=args.steps, 
                stop_at=args.stop_at, 
                batchsize=args.batchsize, 
                use_ref_structure=args.use_ref_structure, 
                device=args.device, 
                )
        
            if args.save_all:
                total_summary = {k: v for k, v in total_summary.items() if k in ALL_ARGS}
            else:
                total_summary = {k: v for k, v in total_summary.items() if k in TOP_ARGS}

            assert all(len(v) == len(total_summary["top_cc"]) for v in total_summary.values()), [f"{k}: {len(v)} == {len(total_summary['top_cc'])}" for k, v in total_summary.items()]
            
            if args.use_ref_structure:
                half_size = int(len(total_summary["top_cc"]) / 2)
                total_summary = {
                    new_k: new_v
                    for k, v in total_summary.items()
                    for new_k, new_v in [(k, v[:half_size]), (k + "_pdb", v[half_size:])]
                }
            
            assert len(batch) == len(total_summary["top_cc"]), f"{len(batch)} != {len(total_summary['top_cc'])}"
            
            if len(total_summary) > 0:
                for i in range(len(batch)):
                    sub_summary = {k: v[i][None,...] for k, v in total_summary.items()}
                    np.savez_compressed(os.path.join(batch[i], "predictions", f"{args.name}{suffix}.npz"), **sub_summary)
                    save_args_to_file(args, os.path.join(batch[i], "predictions", f"{args.name}{suffix}.json"))
        