#!/bin/bash
#SBATCH --job-name=merge_sac_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1            # we manage parallelism manually
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

module purge
module load CUDA/12.4.0
source ~/.bashrc
conda activate componet

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ALGORITHM="sequential-merge"
SEEDS=(0 1 2 3)
SCRIPT=/data/home/co/coimd/componet/experiments/meta-world/run_experiments.py

# Launch 8 runs, 2 per GPU
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU_ID=$(( i % 4 ))

    echo "Launching seed $SEED on GPU $GPU_ID"

    LOGFILE="logs_seed${SEED}_${ALGORITHM}_enhanced.out"
    {
        echo "Log for seed $SEED started at: $(date)"
        echo "GPU: $GPU_ID"
        echo ""
    } >> "$LOGFILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 $SCRIPT \
        --algorithm $ALGORITHM \
        --seed $SEED \
        >> "$LOGFILE" 2>&1 &
done


wait  # wait for all 8 background jobs

echo "=========================================="
echo "All runs completed at: $(date)"
echo "=========================================="
