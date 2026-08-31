#!/bin/sh
#SBATCH --job-name=build_benchmark
#SBATCH --output=build_%j.log
#SBATCH --error=build_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
#SBATCH --partition=dss

# loading necessary modules and activating the virtual environment
module load python/3.10.8_10gcc
cd /lustre/home/smirra/conspiracy_benchmark
source venv/bin/activate

# execution of the build script
echo "building the benchmark (with 32GB of RAM)..."
python scripts/04_build_benchmark.py --split-strategy preserve_original_splits
echo "done building the benchmark."