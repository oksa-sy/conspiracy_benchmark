#!/bin/sh
#SBATCH --job-name=ctbert_train
#SBATCH --output=%x_%A_%a.log
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --array=0-6

# ==========================================
# CLUSTER-SPECIFIC SETTINGS (MODIFY THESE!)
# ==========================================
# 1. Set your partition (e.g., compute, gpu)
# #SBATCH --partition=YOUR_PARTITION_NAME
#
# 2. Hardware requests (CPUs, Mem, and GPUs)
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=32G
# # Uncomment below if your cluster has GPUs:
# #SBATCH --gpus=1
# ==========================================

# Load your specific python module here (if required by your HPC)
# module load python/3.x.x

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

DATASETS=("voat" "ccn" "semeval" "loco" "coco" "donald" "twitter_conspiracy")
TARGET_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python src/modeling/transformer.py $TARGET_DATASET