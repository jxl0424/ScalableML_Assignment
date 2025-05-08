#!/bin/bash
#SBATCH --job-name=Question1 
#SBATCH --time=00:20:00 
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --output=./Output/Q1_Output.txt

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ljingxiang1@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2024.02-1
source activate myspark

echo "Job started at: $(date)"
start=$(date +%s)

spark-submit ./question1.py

end=$(date +%s)
runtime=$((end-start))
echo "Job ended at: $(date)"
echo "Execution time: $runtime seconds"
