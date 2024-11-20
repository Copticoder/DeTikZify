#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100
#SBATCH -C 80g
#SBATCH --time=3-23:59:59
#SBATCH --mem-per-cpu=4G
# Set number of tasks to run
# Set the number of CPU cores for each task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

python generate_captions.py
