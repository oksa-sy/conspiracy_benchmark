#!/bin/sh
#SBATCH --job-name=ctbert_train
#SBATCH --output=%x_%A_%a.log
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=AMD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-6%1
## set the email address for notifications
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=END,FAIL

# loading necessary modules
module load python/3.10.8_10gcc

# moving to the submission directory (user-independent)
cd $SLURM_SUBMIT_DIR

# activating the virtual environment
source venv/bin/activate

# defining the list of target datasets 
DATASETS=("voat" "ccn" "semeval" "loco" "coco" "donald" "twitter_conspiracy")

# extracting the target dataset based on the SLURM array index (0 to 4)
TARGET_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# setting the number of threads for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# starting the training process
echo "starting training process for CT-BERT model on the $TARGET_DATASET dataset using $SLURM_CPUS_PER_TASK core CPU (AMD EPYC)..."
python src/modeling/transformer.py $TARGET_DATASET