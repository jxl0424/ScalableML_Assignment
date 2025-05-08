from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from time import time

start = time()
# Initialize Spark session with memory configuration
spark = SparkSession.builder \
        .appName("Assignment Question 3A") \
        .config("spark.local.dir", "/mnt/parscratch/users/acp24lj") \
        .config("spark.driver.memory", "7g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.javaOptions", "-Xss32m") \
        .config("spark.driver.javaOptions", "-Xss32m") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") 

print("=======================Start of Question 3A=====================")
# Use paths to the training and test datasets
train_path = "./Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv"
test_path = "./Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv"

# Read CSV from the specified paths using Spark builtin read csv functions
train_df = spark.read.csv(train_path, header=False, inferSchema=True)
test_df = spark.read.csv(test_path, header=False, inferSchema=True)

# The first 128 columns are features, and the last column is the label
# Extract feature and label columns and store in variables
num_columns = len(train_df.columns)
feature_col_names = train_df.columns[:-1]  # All except last
label_col = train_df.columns[-1]           # Last column

# Display information for training and test sets
# Using count() to get the number of rows in each dataset
print("Dataset information:")
print(f"Training set size: {train_df.count()} rows")
print(f"Test set size: {test_df.count()} rows")

# Check class distribution
print("\nOriginal class distribution:")
class_distribution = train_df.groupBy(label_col).count().collect()
for row in class_distribution:
    print(f"Class {row[label_col]}: {row['count']} records")

# ============== Stratified Sampling (1% of the dataset) =========================
# Step 1 - Calculate fractions for each class (1% sampling)
# Sample size is set to 0.01 for 1% sampling
sample_size = 0.01 
fractions = {row[label_col]: sample_size for row in class_distribution}

# Perform stratified sampling using sampleBy()
# This will create a balanced(close to balance) sample of the dataset
balanced_sample = train_df.sampleBy(label_col, fractions, seed=31891)

# Step 2 - Convert labels (-1 to 0)
# Since the label column contains -1 and 1,
# Converting -1 to 0 for clarity and standardization
# Most ML libraries expect binary labels to be 0 and 1
balanced_sample = balanced_sample.withColumn(label_col, F.when(F.col(label_col) == -1, 0).otherwise(F.col(label_col)))
test_df = test_df.withColumn(label_col, F.when(F.col(label_col) == -1, 0).otherwise(F.col(label_col)))

# Caching balanced sample and test dataframes to improve performance because they will be used multiple times in the pipeline later on
balanced_sample.cache()
test_df.cache()

# Display the size of the sampled dataset
print(f"\nSampled dataset size: {balanced_sample.count()} rows ({balanced_sample.count()/train_df.count()*100:.2f}% of original)")
print("Sampled class distribution:")
sampled_distribution = balanced_sample.groupBy(label_col).count().collect()
for row in sampled_distribution:
    print(f"Class {row[label_col]}: {row['count']} records")

# Step 3 - Assemble features into a single vector column
# This part is essential for ML algorithms in Spark because they expect a single vector column for features
assembler = VectorAssembler(inputCols=feature_col_names, outputCol="assem_features")

#================================Random Forest Classifier=================================
# Step 4 - Create RandomForestClassifier with default parameters but with a seed of last 5 digit of UCard for reproducibility
rf = RandomForestClassifier(labelCol=label_col, featuresCol="assem_features", seed=31891)

# Create the pipeline for the assembler and the Random Forest model
rf_pipeline = Pipeline(stages=[assembler, rf])

# Step 5 - Define a parameter grid with 3 hyperparameters and 3 values for hyperparameter tuning
rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 15, 20]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.maxBins, [16, 32, 64]) \
    .build()

# Initialize evaluators for AUC and accuracy
auc_evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")

# Initiate a 3-fold cross-validation for hyperparameter tuning
rf_crossval = CrossValidator(estimator=rf_pipeline,
                          estimatorParamMaps=rf_paramGrid,
                          evaluator=auc_evaluator,
                          numFolds=3,
                          seed=31891)

# Step 6 - Run cross-validation on the balanced sample
print("\nStarting cross-validation for Random Forest hyperparameter tuning...")
best_rf_model = rf_crossval.fit(balanced_sample)
# Extract the best rf model from the pipeline cross-validation
rf_model = best_rf_model.bestModel.stages[-1]  

# Extract the best model hyperparameters and print them
print("\nBest Random Forest parameters:")
print(f"Number of Trees: {rf_model.getNumTrees}")
print(f"Max Depth: {rf_model.getMaxDepth()}")
print(f"Max Bins: {rf_model.getMaxBins()}")

# Step 7: Evaluate the final model on test data
print("\nEvaluating Random Forest model on test data")
rf_prediction = best_rf_model.transform(test_df)
rf_auc = auc_evaluator.evaluate(rf_prediction)
rf_acc = accuracy_evaluator.evaluate(rf_prediction)
print(f"Final Random Forest model AUC-ROC on test data: {rf_auc:.4f}")
print(f"Final Random Forest model Accuracy on test data: {rf_acc:.4f}")

#============================================== Gradient Boosted Trees Classifier =================================
# Step 1 - Create GradientBoostedTreesClassifier
gbt = GBTClassifier(labelCol=label_col, featuresCol="assem_features", seed=31891)

# Step 2 - Create the pipeline for the assembler and the GBT model
gbt_pipeline = Pipeline(stages=[assembler, gbt])

# Step 3 - Define a parameter grid with 3 hyperparameters and 3 values for hyperparameter tuning
gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 8, 10]) \
    .addGrid(gbt.maxIter, [10, 20, 50]) \
    .addGrid(gbt.stepSize, [0.01, 0.1, 0.2]) \
    .build()

# Initiate a 3-fold cross-validation for hyperparameter tuning
gbt_crossval = CrossValidator(estimator=gbt_pipeline,
                          estimatorParamMaps=gbt_paramGrid,
                          evaluator=auc_evaluator,
                          numFolds=3,
                          seed=31891)

# Step 4 - Run cross-validation on the balanced sample
print("\nStarting cross-validation for Gradient Boosted Trees hyperparameter tuning...")
best_gbt_model = gbt_crossval.fit(balanced_sample)
gbt_model = best_gbt_model.bestModel.stages[-1]  # Extract the GBT model from the pipeline

print("\nBest Gradient Boosting parameters:")
print(f"Max Depth: {gbt_model.getMaxDepth()}")
print(f"Max Iterations: {gbt_model.getMaxIter()}")
print(f"Step Size (Learning Rate): {gbt_model.getStepSize()}")

# Step 5 - Evaluate the final model on test data
print("\nEvaluating model on test data...")
gbt_prediction = best_gbt_model.transform(test_df)
gbt_auc = auc_evaluator.evaluate(gbt_prediction)
gbt_acc = accuracy_evaluator.evaluate(gbt_prediction)
print(f"Final Gradient Boosted Trees model AUC-ROC on test data: {gbt_auc:.4f}")
print(f"Final Gradient Boosted Trees model Accuracy on test data: {gbt_acc:.4f}")

#============================== (Shallow) Neural Network Classifier ==============================
# Initialise standardScaler for feature scaling
scaler = StandardScaler(inputCol="assem_features", outputCol="features", 
                        withStd=True, withMean=True)

# Create a shallow neural network with MultilayerPerceptronClassifier
# Number of input features is the number of columns in the dataset minus 1 (label column)
num_features = len(feature_col_names)
num_classes = 2  

# Function to define different layer configurations to try
def create_layers(hidden_layer_size):
    # [input layer, hidden layer, output layer]
    return [num_features, hidden_layer_size, num_classes]

# Step 1 - Create MultilayerPerceptronClassifier with default parameters but with a seed of last 5 digit of UCard for reproducibility
mlp = MultilayerPerceptronClassifier(labelCol=label_col, featuresCol="features", seed=31891)

# Create the pipeline for the assembler, scaler, and MLP model
mlp_pipeline = Pipeline(stages=[assembler, scaler, mlp])

# Step 2 - Define a parameter grid for hyperparameter tuning
mlp_paramGrid = ParamGridBuilder() \
    .addGrid(mlp.layers, [create_layers(64),create_layers(32),create_layers(16)]) \
    .addGrid(mlp.maxIter, [10, 20, 50]) \
    .addGrid(mlp.blockSize, [16, 32, 64]) \
    .build()

# Initiate a 3-fold cross-validation for hyperparameter tuning
mlp_crossval = CrossValidator(estimator=mlp_pipeline,
                          estimatorParamMaps=mlp_paramGrid,
                          evaluator=auc_evaluator,
                          numFolds=3,
                          seed=31891)

# Step 3 - Run cross-validation on the balanced sample
print("\nStarting cross-validation for neural network hyperparameter tuning...")
best_mlp_model = mlp_crossval.fit(balanced_sample)
mlp_model = best_mlp_model.bestModel.stages[-1]  # Extract the MLP model from the pipeline

print("\nBest Neural Network parameters:")
print(f"Layers: {mlp_model.getLayers()}")
print(f"Max Iterations: {mlp_model.getMaxIter()}")
print(f"Block Size: {mlp_model.getBlockSize()}")

# Step 4 - Evaluate the final model on test data
print("\nEvaluating neural network model on test data...")
mlp_prediction = best_mlp_model.transform(test_df)
mlp_auc = auc_evaluator.evaluate(mlp_prediction)
mlp_acc = accuracy_evaluator.evaluate(mlp_prediction)
print(f"Final neural network model AUC-ROC on test data: {mlp_auc:.4f}")
print(f"Final neural network model Accuracy on test data: {mlp_acc:.4f}")

end = time()

# Stop Spark session
spark.stop()

