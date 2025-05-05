#!/bin/bash
#SBATCH --job-name=Question3A 
#SBATCH --mem=10G 
#SBATCH --cpus-per-task=10
#SBATCH --output=./Output/Q3A_Output.txt

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ljingxiang1@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2024.02-1
source activate myspark

spark-submit --driver-memory 2g --executor-memory 8g --master local[8] ./question3.py
