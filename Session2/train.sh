#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Load necessary modules
module load Python/3.8.6-GCCcore-10.2.0
module load Anaconda3/2024.02-1

source activate base

jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''

