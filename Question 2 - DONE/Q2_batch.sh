#!/bin/bash
#SBATCH --job-name=Question2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --output=./Output/Q2_Output.txt

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ljingxiang1@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2024.02-1
source activate myspark

echo "Job started at: $(date)"
start=$(date +%s)

spark-submit --driver-memory 5g --executor-memory 5g --master local[4] question2.py

end=$(date +%s)
runtime=$((end-start))
echo "Job ended at: $(date)"
echo "Execution time: $runtime seconds"

