#!/bin/bash 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=24G 
#SBATCH --job-name=test 
#SBATCH --time=360

LOG_DIR="$PWD/logs"
mkdir -p $LOG_DIR

#SBATCH --output=$LOG_DIR/output_%x_%j.out
#SBATCH --error=$LOG_DIR/error_%x_%j.err

  
module load anaconda 
source activate base # ( this would activate your conda base env )
conda activate sc4001
python -u train.py --name=180325-expt1 --config=configs/vit_head_tuned_v1.yaml --gpus 0 --max_epochs=100 --batch_size=16 --grad_acc=2
