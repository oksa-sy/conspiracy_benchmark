#!/bin/sh
#SBATCH --job-name=ctbert_train
#SBATCH --output=%x_%A_%a.log
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=dss
#SBATCH --array=0-4
## set the email address for notifications
#SBATCH --mail-user=your.email@example.it
#SBATCH --mail-type=END,FAIL

# loading necessary modules
module load python/3.10.8_10gcc

# moving to the submission directory (user-independent)
cd $SLURM_SUBMIT_DIR

# activating the virtual environment
source venv/bin/activate

# defining the list of target datasets (Semeval is currently excluded)
DATASETS=("voat" "ccn" "semeval" "loco" "coco" "donald")

# extracting the target dataset based on the SLURM array index (0 to 4)
TARGET_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# starting the training process
echo "starting training process for CT-BERT model on the $TARGET_DATASET dataset..."
python src/modeling/transformer.py $TARGET_DATASET