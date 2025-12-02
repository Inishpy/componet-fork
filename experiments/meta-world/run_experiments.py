import subprocess
import argparse
import random
from tasks import tasks
import logging
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "simple",
            "componet",
            "finetune",
            "from-scratch",
            "prognet",
            "packnet",
            "sequential-merge",
            "diffusion"
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-run", default=False, action="store_true")

    parser.add_argument("--start-mode", default=2, type=int, required=False)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to use")
    return parser.parse_args()


args = parse_args()

# Create a timestamped subdirectory for this experiment run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_root = f"agents/{timestamp}"
os.makedirs(save_root, exist_ok=True)

modes = list(range(20)) if args.algorithm != "simple" else list(range(10))

# NOTE: If the algoritm is not `simple  packnet prognet`, it always should start from the second task
if args.algorithm not in ["simple", "packnet", "prognet", "sequential-merge"] and args.start_mode == 0:
    start_mode = 1
else:
    start_mode = args.start_mode

run_name = (
    lambda task_id: f"task_{task_id}__{args.algorithm if task_id > 0 or args.algorithm in ['packnet', 'prognet', 'sequential-merge'] else 'simple'}__run_sac__{args.seed}"
)

first_idx = modes.index(start_mode)
for i, task_id in enumerate(modes[first_idx:]):
    params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed}"
    params += f" --save-dir={save_root}"

    logging.info("task id", i, task_id, "starting training")

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prognet"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" {save_root}/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet"]:
            params += f" --prev-units {save_root}/{run_name(task_id-1)}"
        elif args.algorithm == "sequential-merge":
            if task_id >= 3:
                params += f" --prev-units {save_root}/{run_name(task_id-1)}"

    # Launch experiment
    cmd = f"./run_sac.py {params}"
    print(cmd)

    if not args.no_run:
        res = subprocess.run(cmd.split(" "))
        if res.returncode != 0:
            print(f"*** Process returned code {res.returncode}. Stopping on error.")
            quit(1)

    logging.info("task id", i, task_id, "starting training")
