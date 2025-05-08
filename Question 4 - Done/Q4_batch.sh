#!/bin/bash
#SBATCH --job-name=Question4
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=10
#SBATCH --output=./Output/Q4_Output.txt

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ljingxiang1@sheffield.ac.uk

module load Java/17.0.4
module load Anaconda3/2024.02-1
source activate myspark

echo "Job started at: $(date)"
start=$(date +%s)

spark-submit --driver-memory 10g --executor-memory 10g --master local[10] --conf spark.driver.extraJavaOptions=-Xss32m --conf spark.executor.extraJavaOptions=-Xss32m ./question4.py

end=$(date +%s)
runtime=$((end-start))
echo "Job ended at: $(date)"
echo "Execution time: $runtime seconds"
