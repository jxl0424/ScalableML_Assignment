from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from time import time
import pandas as pd
import gc
import os
import csv
from datetime import datetime

# ================== Sample Size ==================
# Set the sample size for this run (change this manually between runs)
SAMPLE_SIZE = 0.9  # 90% of the dataset

# ================== Initialize Spark ==================
start_time = time()
spark = SparkSession.builder \
        .appName("PUF Classification with Sample Size") \
        .config("spark.local.dir", "/mnt/parscratch/users/acp24lj") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") 

print("============== Start of Question 3b ==============")
print(f"\n========== {SAMPLE_SIZE*100:.1f}% Sample Size ==========")

# ================== Load Data ==================
# Use paths to the training and test datasets
train_path = "./Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv"
test_path = "./Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv"

# Read CSV files
train_df = spark.read.csv(train_path, header=False, inferSchema=True)
test_df = spark.read.csv(test_path, header=False, inferSchema=True)

# Extract feature and label columns
feature_col_names = train_df.columns[:-1]  # All columns except the last one
label_col = train_df.columns[-1]           # Last column is the label

# Display dataset information
print(f"Training set size: {train_df.count()} rows")
print(f"Test set size: {test_df.count()} rows")

# Check class distribution
print("\nOriginal class distribution:")
class_distribution = train_df.groupBy(label_col).count().collect()
for row in class_distribution:
    print(f"Class {row[label_col]}: {row['count']} records")

# Convert test labels from -1 to 0 for binary classification
test_df = test_df.withColumn(label_col, F.when(F.col(label_col) == -1, 0).otherwise(F.col(label_col)))
test_df.cache()  # Cache test dataframe as it will be used repeatedly

# ================== Set Model Parameters ==================
# Best model parameters from question 3a
best_rf_params = {
    'numTrees': 20,
    'maxDepth': 15,
    'maxBins': 16,
}

best_gbt_params = {
    'maxIter': 5,
    'maxDepth': 10,
    'stepSize': 0.2,
}

best_mlp_params = {
    'layers': [128, 32, 2],
    'maxIter': 50,
    'blockSize': 64
}

# ================== Prepare Model Components ==================
# Prepare feature assembler and evaluators
assembler = VectorAssembler(inputCols=feature_col_names, outputCol="assem_features")

# Evaluators for model performance
auc_evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")

# ================== Create Model Pipelines ==================
# Random Forest pipeline
rf = RandomForestClassifier(
    labelCol=label_col, 
    featuresCol="assem_features", 
    seed=31891, 
    **best_rf_params
)
pipeline_rf = Pipeline(stages=[assembler, rf])

# Gradient Boosted Trees pipeline
gbt = GBTClassifier(
    labelCol=label_col, 
    featuresCol="assem_features", 
    seed=31891, 
    **best_gbt_params
)
pipeline_gbt = Pipeline(stages=[assembler, gbt])

# Multi-layer Perceptron pipeline
mlp = MultilayerPerceptronClassifier(
    labelCol=label_col, 
    featuresCol="assem_features", 
    seed=31891, 
    **best_mlp_params
)
pipeline_mlp = Pipeline(stages=[assembler, mlp])

# ================== Prepare Sampled Dataset ==================
# Calculate fractions for stratified sampling to maintain class balance
fractions = {row[label_col]: SAMPLE_SIZE for row in class_distribution}

# Perform stratified sampling
balanced_sample = train_df.sampleBy(label_col, fractions, seed=31891)

# Convert labels from -1 to 0 for binary classification
balanced_sample = balanced_sample.withColumn(
    label_col, 
    F.when(F.col(label_col) == -1, 0).otherwise(F.col(label_col))
)

# Cache the sample for better performance
balanced_sample.cache()
sample_count = balanced_sample.count()

print(f"\nSampled dataset size: {sample_count} rows ({SAMPLE_SIZE*100:.1f}% of original)")
print("Sampled class distribution:")
sampled_distribution = balanced_sample.groupBy(label_col).count().collect()
for row in sampled_distribution:
    print(f"Class {row[label_col]}: {row['count']} records")

# ================== Run Models ==================
results = {
    'sample_size': SAMPLE_SIZE,
    'sample_count': sample_count,
}

# -------------------- Random Forest --------------------
print("\nTraining Random Forest model...")
rf_start = time()
rf_model = pipeline_rf.fit(balanced_sample)

# Get predictions and evaluate
rf_predictions = rf_model.transform(test_df)
rf_auc = auc_evaluator.evaluate(rf_predictions)
rf_accuracy = accuracy_evaluator.evaluate(rf_predictions)

rf_time = time() - rf_start

print(f"RF AUC: {rf_auc:.4f}")
print(f"RF Accuracy: {rf_accuracy:.4f}")
print(f"RF Training Time: {rf_time:.2f} seconds")

# Store results
results['rf_accuracy'] = rf_accuracy
results['rf_auc'] = rf_auc
results['rf_time'] = rf_time

# -------------------- Gradient Boosted Trees --------------------
print("\nTraining GBT model...")
gbt_start = time()
gbt_model = pipeline_gbt.fit(balanced_sample)

# Get predictions and evaluate
gbt_predictions = gbt_model.transform(test_df)
gbt_auc = auc_evaluator.evaluate(gbt_predictions)
gbt_accuracy = accuracy_evaluator.evaluate(gbt_predictions)

gbt_time = time() - gbt_start

print(f"GBT AUC: {gbt_auc:.4f}")
print(f"GBT Accuracy: {gbt_accuracy:.4f}")
print(f"GBT Training Time: {gbt_time:.2f} seconds")

# Store results
results['gbt_accuracy'] = gbt_accuracy
results['gbt_auc'] = gbt_auc
results['gbt_time'] = gbt_time

# -------------------- Multi-layer Perceptron --------------------
print("\nTraining MLP model...")
mlp_start = time()

mlp_model = pipeline_mlp.fit(balanced_sample)

# Get predictions and evaluate
mlp_predictions = mlp_model.transform(test_df)
mlp_auc = auc_evaluator.evaluate(mlp_predictions)
mlp_accuracy = accuracy_evaluator.evaluate(mlp_predictions)

mlp_time = time() - mlp_start

print(f"MLP AUC: {mlp_auc:.4f}")
print(f"MLP Accuracy: {mlp_accuracy:.4f}")
print(f"MLP Training Time: {mlp_time:.2f} seconds")

# Store results
results['mlp_accuracy'] = mlp_accuracy
results['mlp_auc'] = mlp_auc
results['mlp_time'] = mlp_time

# ================== Save Results ==================
# Calculate total time for this sample size
total_time = rf_time + gbt_time + mlp_time
print(f"\nTotal execution time for {SAMPLE_SIZE*100:.1f}% sample: {total_time:.2f} seconds")

# CSV filename for results
csv_filename = "puf_model_scaling_results.csv"

# Check if file exists to decide whether to write headers
file_exists = os.path.isfile(csv_filename)


# Define field names for CSV
fieldnames = [
    'sample_size', 'sample_count', 
    'rf_accuracy', 'rf_auc', 'rf_time',
    'gbt_accuracy', 'gbt_auc', 'gbt_time',
    'mlp_accuracy', 'mlp_auc', 'mlp_time'
]

# Write to CSV (append mode)
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write header only if file doesn't exist
    if not file_exists:
        writer.writeheader()
    
    writer.writerow(results)

print(f"Results appended to {os.path.abspath(csv_filename)}")

# ================== Cleanup ==================
# Unpersist datasets
balanced_sample.unpersist()
test_df.unpersist()
spark.catalog.clearCache()
gc.collect()

# Total execution time
end_time = time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds")

# Stop Spark session
spark.stop()