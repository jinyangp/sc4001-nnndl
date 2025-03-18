#!/bin/bash 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=24G 
#SBATCH --job-name=test 
#SBATCH --time=360
#SBATCH --output=output_%x_%j.out 
#SBATCH --error=error_%x_%j.err 

module load anaconda 
source activate base # ( this would activate your conda base env )
conda activate sc4001
pip list