#!/bin/sh
#SBATCH --job-name=ctbert_train
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=dss
#SBATCH --mail-user=mirra.2111616@studenti.uniroma1.it
#SBATCH --mail-type=END,FAIL

# loading necessary modules
module load python/3.10.8_10gcc

# moving to the project directory
cd /lustre/home/smirra/conspiracy_benchmark

# activating the virtual environment
source /lustre/home/smirra/conspiracy_benchmark/venv/bin/activate

# starting the training process
# voat
echo "starting training process for CT-BERT model on the VOAT dataset..."
python src/modeling/transformer.py voat

# semeval
# echo "starting training process for CT-BERT model on the Semeval dataset..."
# python src/modeling/transformer.py semeval

# ccn
echo "starting training process for CT-BERT model on the CCN dataset..."
python src/modeling/transformer.py ccn

# loco
echo "starting training process for CT-BERT model on the LOCO dataset..."
python src/modeling/transformer.py loco

# coco
echo "starting training process for CT-BERT model on the COCO dataset..."
python src/modeling/transformer.py coco

# donald
echo "starting training process for CT-BERT model on the DONALD dataset..."
python src/modeling/transformer.py donald