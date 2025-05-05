from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, countDistinct
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.sql.types import IntegerType, DoubleType
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from time import time

start = time()
# Initialize Spark Session with 4 cores on Stanage
spark = SparkSession.builder \
.master("local[4]") \
.appName("Assignment Question 2") \
.config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR") 

# Load data
print("Loading dataset...")
df_pandas = kagglehub.load_dataset(
KaggleDatasetAdapter.PANDAS,
"brandao/diabetes",
"diabetic_data.csv" 
)

df = spark.createDataFrame(df_pandas)

# Define medication features
medication_features = [
'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
'miglitol', 'troglitazone', 'tolazamide', 'examide',
'citoglipton', 'insulin', 'glyburide-metformin', 
'glipizide-metformin', 'glimepiride-pioglitazone',
'metformin-rosiglitazone', 'metformin-pioglitazone'
]

# Process target variables
# 1. Convert readmitted to binary (0 for "NO", 1 for others)
df = df.withColumn("readmitted", 
when(col("readmitted") == "NO", 0).otherwise(1))
df = df.withColumn("readmitted", col("readmitted").cast(IntegerType()))

# 2. Ensure numeric targets are properly typed
df = df.withColumn("time_in_hospital", col("time_in_hospital").cast(DoubleType()))
df = df.withColumn("num_lab_procedures", col("num_lab_procedures").cast(DoubleType()))

# Cache the DataFrame for better performance
df.cache()

# Define the sampling fractions for 80/20 split
fractions = {0: 0.8, 1: 0.8}

# Split data into training and test sets with stratification
train_df = df.sampleBy("readmitted", fractions=fractions, seed=31891).cache()
test_df = df.subtract(train_df).cache()

# Print dataset statistics
print("Dataset statistics:")
print(f"Total: {df.count()}")
print(f"Train: {train_df.count()}")
print(f"Test:  {test_df.count()}")

print("\nClass distribution in splits:")
print("Train:")
train_df.groupBy("readmitted").count().show()
print("Test:")
test_df.groupBy("readmitted").count().show()

# Create StringIndexers for each medication feature
indexers = [StringIndexer(inputCol=feature, outputCol=f"{feature}_index")
for feature in medication_features]

# Create OneHotEncoders for each indexed feature
encoders = [OneHotEncoder(inputCol=f"{feature}_index", outputCol=f"{feature}_vec", dropLast=False)
for feature in medication_features]

# Create a VectorAssembler to combine all encoded features
encoded_features = [f"{feature}_vec" for feature in medication_features]
assembler = VectorAssembler(inputCols=encoded_features, outputCol="medication_features_vector")

# Build the pipeline with all stages
pipeline_stages = indexers + encoders + [assembler]
pipeline = Pipeline(stages=pipeline_stages)

# Fit the pipeline to the training data
pipeline_model = pipeline.fit(train_df)

# Transform both training and test data
train_df_transformed = pipeline_model.transform(train_df).cache()
test_df_transformed = pipeline_model.transform(test_df).cache()

# Show a sample of the processed data
print("Sample of processed features:")
train_df_transformed.select("medication_features_vector", "readmitted").show(5, truncate=True)

# Define parameter grid values for cross-validation
reg_params = [0.001, 0.01, 0.1, 1, 10, 100]
elastic_net_params = [0, 0.2, 0.5, 0.8, 1]

#======================= Poisson Regression =======================
# Train a Poisson Regression model
print("=============Training Poisson Regression model starts======================")
poisson_reg = GeneralizedLinearRegression(
    family="poisson",
    link="log",
    featuresCol="medication_features_vector",
    labelCol="time_in_hospital",
    maxIter=100,
)

# Create parameter grid
poisson_param_grid = ParamGridBuilder() \
    .addGrid(poisson_reg.regParam, reg_params) \
    .build()

# Create evaluator
poisson_evaluator = RegressionEvaluator(
    labelCol="time_in_hospital", 
    predictionCol="prediction",
    metricName="rmse"  # Using RMSE for regression
)

# Set up cross-validator
poisson_cv = CrossValidator(
    estimator=poisson_reg,
    estimatorParamMaps=poisson_param_grid,
    evaluator=poisson_evaluator,
    numFolds=5,
    seed=31891,
    parallelism=4
)

# Run cross-validation
print("Running Poisson Regression cross-validation")
poisson_cv_model = poisson_cv.fit(train_df_transformed)

# Get best parameters and model
poisson_best_model = poisson_cv_model.bestModel
poisson_best_reg_param = poisson_best_model.getRegParam()

print(f"Best Poisson Regression parameters:")
print(f"Best Poisson Regression regParam: {poisson_best_reg_param}")

# Extract metrics for validation curve
poisson_avg_metrics = poisson_cv_model.avgMetrics
poisson_std_metrics = poisson_cv_model.stdMetrics
print(f"Average metrics: {poisson_avg_metrics}")
print(f"Standard deviation of metrics: {poisson_std_metrics}")

# Plot validation curve for Poisson regression
plt.figure(figsize=(10, 6))
plt.errorbar(reg_params, poisson_avg_metrics, yerr=poisson_std_metrics, fmt='-o', label='Poisson Regression')
plt.xscale('log')
plt.xlabel('regParam (log scale)')
plt.ylabel('RMSE')
plt.title('Poisson Regression Validation Curve')
plt.legend()
plt.savefig('poisson_validation_curve.png')
plt.close()


# Train a Logistic Regression model for readmission prediction
print("\n================Training Logistic Regression with L2 Regularisation model starts==================")
lr_l2 = LogisticRegression(
featuresCol="medication_features_vector",
labelCol="readmitted",
maxIter=100,
family="binomial",
elasticNetParam=0.0
)

# Create parameter grid for cross-validation
lr_l2_param_grid = ParamGridBuilder() \
.addGrid(lr_l2.regParam, reg_params) \
.build()

# Create evaluator for logistic regression
lr_evaluator = MulticlassClassificationEvaluator(
labelCol="readmitted", 
predictionCol="prediction",
metricName="accuracy"  
)

# Set up cross-validator
lr_l2_cv = CrossValidator(
estimator=lr_l2,
estimatorParamMaps=lr_l2_param_grid,
evaluator=lr_evaluator,
numFolds=5,
seed=31891,
parallelism=4
)

# Run cross-validation
print("Running Logistic Regression cross-validation")
lr_l2_cv_model = lr_l2_cv.fit(train_df_transformed)

# Get best parameters and model
best_lr_l2_model = lr_l2_cv_model.bestModel
best_lr_l2_reg_param = best_lr_l2_model.getRegParam()

print(f"Best Logistic Regression with L2 Regularisation parameters:")
print(f"Best regParam value: {best_lr_l2_reg_param}")

# Calculate standard deviation of metrics
lr_l2_avg_metrics = lr_l2_cv_model.avgMetrics
lr_l2_std_metrics = np.array(lr_l2_cv_model.stdMetrics)
print(f"Average metrics: {lr_l2_avg_metrics}")
print(f"Standard deviation of metrics: {lr_l2_std_metrics}")

# Plot validation curve for Logistic regression with L2 regularization
plt.figure(figsize=(10, 6))
plt.errorbar(reg_params, lr_l2_avg_metrics, yerr=lr_l2_std_metrics, fmt='-o', label='Logistic Regression L2')
plt.xscale('log')
plt.xlabel('regParam (log scale)')
plt.ylabel('Accuracy')
plt.title('Logistic Regression with L2 Regularisation Validation Curve')
plt.legend()
plt.savefig('logistic_reg_l2_curve.png')

# ======================== Logistic Regression with Elastic Net=======================
# Train a Logistic Regression model with Elastic Net
print("\n=================Training Logistic Regression model with Elastic Net==================")
lr_EN = LogisticRegression(
featuresCol="medication_features_vector",
labelCol="readmitted",
maxIter=100,
family="binomial",
)

# Create parameter grid for cross-validation
lr_EN_param_grid = ParamGridBuilder() \
    .addGrid(lr_EN.regParam, reg_params) \
    .addGrid(lr_EN.elasticNetParam, elastic_net_params) \
    .build()

# Create evaluator for logistic regression with Elastic Net
lr_EN_cv = CrossValidator(
    estimator=lr_EN,
    estimatorParamMaps=lr_EN_param_grid,
    evaluator=lr_evaluator,
    numFolds=5,
    seed=31891,
    parallelism=4
)

# Run cross-validation
lr_EN_cv_model = lr_EN_cv.fit(train_df_transformed)

# Get best parameters and model
best_lr_EN_model = lr_l2_cv_model.bestModel
best_lr_EN_reg_param = best_lr_l2_model.getRegParam()
best_lr_EN_elastic_net_param = best_lr_EN_model.getElasticNetParam()

print(f"Best Logistic Regression with Elastic Net parameters:")
print(f"Best regParam value: {best_lr_EN_reg_param}")
print(f"Best regParam value: {best_lr_EN_elastic_net_param}")

# Logistic Regression with Elastic Net metrics
lr_EN_avg_metrics = lr_EN_cv_model.avgMetrics
lr_EN_std_metrics = np.array(lr_EN_cv_model.stdMetrics)

print(f"Average metrics: {lr_EN_avg_metrics}")
print(f"Standard deviation of metrics: {lr_EN_std_metrics}")

# Reshape metrics into 2D array (regParams × elasticNetParams)
en_avg_metrics = np.array(lr_EN_avg_metrics).reshape(len(reg_params), len(elastic_net_params))
en_std_metrics = np.array(lr_EN_std_metrics).reshape(len(reg_params), len(elastic_net_params))

colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, j in enumerate(elastic_net_params):
    plt.errorbar(x=np.log10(reg_params),
                 y=en_avg_metrics[:, i],
                 yerr=en_std_metrics[:, i],
                 fmt='-o',
                 color=colors[i],
                 label=f'α={j}')
    
plt.title("Elastic-Net Logistic\nAccuracy vs Regularization")
plt.xlabel("log10(regParam)")
plt.ylabel("Accuracy (with std dev)")
plt.xticks(np.log10(reg_params), reg_params)
plt.legend()
plt.savefig('logistic_reg_EN.png')

# Evaluate the best model on the test set
poisson_predictions = poisson_best_model.transform(test_df_transformed)
lr_l2_predictions = best_lr_l2_model.transform(test_df_transformed)
lr_EN_predictions = best_lr_EN_model.transform(test_df_transformed)

poisson_accuracy = poisson_evaluator.evaluate(poisson_predictions)
lr_l2_accuracy = lr_evaluator.evaluate(lr_l2_predictions)
lr_EN_accuracy = lr_evaluator.evaluate(lr_EN_predictions)

print("Results on test set of all of the best models:")
print(f"Poisson Regression accuracy: {poisson_accuracy:.4f}")
print(f"Logistic Regression with L2 Regularisation accuracy: {lr_l2_accuracy:.4f}")
print(f"Logistic Regression with Elastic Net accuracy: {lr_EN_accuracy:.4f}")

spark.stop()

end = time()
