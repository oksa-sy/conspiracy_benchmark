#!/bin/sh
#SBATCH --job-name=build_benchmark
#SBATCH --output=build_%j.log
#SBATCH --error=build_%j.err
#SBATCH --time=0-01:00:00

# ==========================================
# CLUSTER-SPECIFIC SETTINGS (MODIFY THESE!)
# ==========================================
# 1. Set your partition (e.g., compute, standard)
# #SBATCH --partition=YOUR_PARTITION_NAME
#
# 2. Hardware requests (RAM is critical for large datasets)
#SBATCH --mem=32G
# ==========================================

# Load your specific python module here (if required by your HPC)
# module load python/3.x.x

# moving to the submission directory (user-independent)
cd $SLURM_SUBMIT_DIR

# activating the virtual environment
source venv/bin/activate

# execution of the build script
echo "building the benchmark (with 32GB of RAM)..."
python scripts/04_build_benchmark.py --split-strategy preserve_original_splits
echo "done building the benchmark."