from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from time import time

start = time()
# Initialize Spark session with memory configuration
spark = SparkSession.builder \
        .master("local[10]") \
        .appName("Assignment Question 3A") \
        .config("spark.local.dir", "/mnt/parscratch/users/acp24lj") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") 

# Load data without header
train_path = "./Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv"
test_path = "./Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv"

# Read CSV without header and infer schema
train_df = spark.read.csv(train_path, header=False, inferSchema=True)
test_df = spark.read.csv(test_path, header=False, inferSchema=True)

# Automatically determine feature and label columns
num_columns = len(train_df.columns)
feature_col_names = train_df.columns[:-1]  # All except last
label_col = train_df.columns[-1]           # Last column

# Display dataset information
print("Dataset information:")
print(f"Training set size: {train_df.count()} rows")
print(f"Test set size: {test_df.count()} rows")

# Check class distribution
print("\nOriginal class distribution:")
class_distribution = train_df.groupBy(label_col).count().collect()
for row in class_distribution:
    print(f"Class {row[label_col]}: {row['count']} records")

# ====== Stratified Sampling (1% of the dataset) ======
# Step 1: Calculate fractions for each class (1% sampling)
class_counts = train_df.groupBy(label_col).count().collect()
total_count = train_df.count()
fractions = {row[label_col]: (row['count'] / total_count) * 0.01 for row in class_counts}  # 1% sampling

# Perform stratified sampling
balanced_sample = train_df.sampleBy(label_col, fractions, seed=31891)

# Step 2: Convert labels (-1 to 0)
balanced_sample = balanced_sample.withColumn(label_col, F.when(F.col(label_col) == -1, 0).otherwise(F.col(label_col)))
test_df = test_df.withColumn(label_col, F.when(F.col(label_col) == -1, 0).otherwise(F.col(label_col)))

# Cache the sample to improve performance
balanced_sample.cache()
test_df.cache()

# Display the size of the sampled dataset
print(f"\nSampled dataset size: {balanced_sample.count()} rows ({balanced_sample.count()/train_df.count()*100:.2f}% of original)")
print("Sampled class distribution:")
sampled_distribution = balanced_sample.groupBy(label_col).count().collect()
for row in sampled_distribution:
    print(f"Class {row[label_col]}: {row['count']} records")


# Step 3: Assemble features into a single vector column
assembler = VectorAssembler(inputCols=feature_col_names, outputCol="assem_features")

#================================Random Forest Classifier=================================
# Create RandomForestClassifier
rf = RandomForestClassifier(labelCol=label_col, featuresCol="assem_features", seed=31891)

# Now proceed with a simplified parameter grid
rf_pipeline = Pipeline(stages=[assembler, rf])

# Step 4: Define a reduced parameter grid for hyperparameter tuning
# Using a smaller grid to reduce memory usage
rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.featureSubsetStrategy, [16, 32, 64]) \
    .build()

auc_evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")

# Use fewer folds to reduce memory requirements
rf_crossval = CrossValidator(estimator=rf_pipeline,
                          estimatorParamMaps=rf_paramGrid,
                          evaluator=auc_evaluator,
                          numFolds=3,
                          seed=31891)

# Step 5: Run cross-validation on the balanced sample
print("\nStarting cross-validation for parameter tuning...")
rf_cv_model = rf_crossval.fit(balanced_sample)

# Step 6: Get the best model and its parameters
best_pipeline_model = rf_cv_model.bestModel
best_rf_model = best_pipeline_model.stages[-1]

print("\nBest Random Forest parameters:")
print(f"Number of Trees: {best_rf_model.getNumTrees}")
print(f"Max Depth: {best_rf_model.getMaxDepth()}")
print(f"Max Bins: {best_rf_model.getMaxBins()}")

rf_avg_metrics = rf_cv_model.avgMetrics
print(f"Best cross-validation AUC: {max(rf_avg_metrics):.4f}")

# Step 7: Evaluate the final model on test data
print("\nEvaluating model on test data...")
rf_prediction = best_pipeline_model.transform(test_df)
rf_auc = auc_evaluator.evaluate(rf_prediction)
rf_acc = accuracy_evaluator.evaluate(rf_prediction)
print(f"Final model AUC-ROC on test data: {rf_auc:.4f}")
print(f"Final model Accuracy on test data: {rf_acc:.4f}")

#============================================== Gradient Boosted Trees Classifier =================================
# Create GradientBoostedTreesClassifier
gbt = GBTClassifier(labelCol=label_col, featuresCol="assem_features", seed=31891)

# Create the pipeline
gbt_pipeline = Pipeline(stages=[assembler, gbt])

# Step 4: Define a parameter grid for hyperparameter tuning
gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 8, 10]) \
    .addGrid(gbt.maxIter, [20, 50, 100]) \
    .addGrid(gbt.stepSize, [0.01, 0.1, 0.2]) \
    .build()

# Use 3-fold cross-validation to match the original code
gbt_crossval = CrossValidator(estimator=gbt_pipeline,
                          estimatorParamMaps=gbt_paramGrid,
                          evaluator=auc_evaluator,
                          numFolds=3,
                          seed=31891)

# Step 5: Run cross-validation on the balanced sample
print("\nStarting cross-validation for parameter tuning...")
gbt_cv_model = gbt_crossval.fit(balanced_sample)

# Step 6: Get the best model and its parameters
best_gbt_pipeline_model = gbt_cv_model.bestModel
best_gbt_model = best_gbt_pipeline_model.stages[-1]

print("\nBest Gradient Boosting parameters:")
print(f"Max Depth: {best_gbt_model.getMaxDepth()}")
print(f"Max Iterations: {best_gbt_model.getMaxIter()}")
print(f"Step Size (Learning Rate): {best_gbt_model.getStepSize()}")

gbt_avg_metrics = gbt_cv_model.avgMetrics
print(f"Best cross-validation AUC: {max(gbt_avg_metrics):.4f}")

#============================== (Shallow) Neural Network Classifier ==============================
# Create MultilayerPerceptronClassifier

# Standardize features
scaler = StandardScaler(inputCol="assem_features", outputCol="features", 
                        withStd=True, withMean=True)

# Create a shallow neural network with MultilayerPerceptronClassifier
# Structure: input layer (128) -> hidden layer (variable) -> output layer (2 classes)
num_features = len(feature_col_names)
num_classes = 2  # Binary classification

# Define different layer configurations to try
# We'll try a shallow network with just one hidden layer of varying sizes
def create_layers(hidden_layer_size):
    # [input layer, hidden layer, output layer]
    return [num_features, hidden_layer_size, num_classes]

# Create MultilayerPerceptronClassifier
mlp = MultilayerPerceptronClassifier(labelCol=label_col, featuresCol="features", seed=31891)

# Create the pipeline
mlp_pipeline = Pipeline(stages=[assembler, scaler, mlp])

# Step 5: Define a parameter grid for hyperparameter tuning
mlp_paramGrid = ParamGridBuilder() \
    .addGrid(mlp.layers, [create_layers(64),create_layers(32),create_layers(16)]) \
    .addGrid(mlp.maxIter, [50, 100, 200]) \
    .addGrid(mlp.blockSize, [32, 64, 128]) \
    .build()

# Use 3-fold cross-validation
mlp_crossval = CrossValidator(estimator=mlp_pipeline,
                          estimatorParamMaps=mlp_paramGrid,
                          evaluator=auc_evaluator,
                          numFolds=3,
                          seed=31891)

# Step 6: Run cross-validation on the balanced sample
print("\nStarting cross-validation for neural network parameter tuning...")
mlp_cv_model = mlp_crossval.fit(balanced_sample)

# Step 7: Get the best model and its parameters
best_mlp_pipeline_model = mlp_cv_model.bestModel
best_mlp_model = best_mlp_pipeline_model.stages[-1]

print("\nBest Neural Network parameters:")
print(f"Layers: {best_mlp_model.getLayers()}")
print(f"Max Iterations: {best_mlp_model.getMaxIter()}")
print(f"Step Size (Learning Rate): {best_mlp_model.getStepSize()}")

mlp_avg_metrics = mlp_cv_model.avgMetrics
print(f"Best cross-validation AUC: {max(mlp_avg_metrics):.4f}")

# Step 8: Evaluate the final model on test data
print("\nEvaluating neural network model on test data...")
mlp_prediction = best_mlp_pipeline_model.transform(test_df)
mlp_auc = auc_evaluator.evaluate(mlp_prediction)
mlp_acc = accuracy_evaluator.evaluate(mlp_prediction)
print(f"Final neural network model AUC-ROC on test data: {mlp_auc:.4f}")
print(f"Final neural network model Accuracy on test data: {mlp_acc:.4f}")

# Stop Spark session
spark.stop()

end = time()