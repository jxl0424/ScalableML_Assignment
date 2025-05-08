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

echo "Job started at: $(date)"
start=$(date +%s)

spark-submit --driver-memory 5g --executor-memory 5g --master local[10] ./question3a.py

end=$(date +%s)
runtime=$((end-start))
echo "Job ended at: $(date)"
echo "Execution time: $runtime seconds"
