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
    .appName("Assignment Question 2") \
    .config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") 

# Load data
print("\n==================Starting Question 2===================")
print("Loading dataset...")
df_pandas = kagglehub.load_dataset(
KaggleDatasetAdapter.PANDAS,
"brandao/diabetes",
"diabetic_data.csv" 
)

# Step 1 - Convert the dataset to a Spark DataFrame
df = spark.createDataFrame(df_pandas)

# Step 2 - Define medication features and preprocessing
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
# Convert readmitted to binary (0 for "NO", 1 for >30/<30)
df = df.withColumn("readmitted", 
when(col("readmitted") == "NO", 0).otherwise(1))
df = df.withColumn("readmitted", col("readmitted").cast(IntegerType()))

# 2. Ensure numeric targets are properly typed
df = df.withColumn("time_in_hospital", col("time_in_hospital").cast(DoubleType()))

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
pipeline = Pipeline(stages= pipeline_stages)

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

# Create parameter grid for poisson regression cross-validation
poisson_param_grid = ParamGridBuilder() \
    .addGrid(poisson_reg.regParam, reg_params) \
    .build()

# Create poisson regression evaluator to get the RMSE
poisson_evaluator = RegressionEvaluator(
    labelCol="time_in_hospital", 
    predictionCol="prediction",
    metricName="rmse"  
)

# Set up cross-validator with 3 folds
poisson_cv = CrossValidator(
    estimator=poisson_reg,
    estimatorParamMaps=poisson_param_grid,
    evaluator=poisson_evaluator,
    numFolds=3,
    seed=31891,
    parallelism=4
)

# Run cross-validation
print("Running Poisson Regression cross-validation...")
poisson_cv_model = poisson_cv.fit(train_df_transformed)

# Get best parameters and model
poisson_best_model = poisson_cv_model.bestModel
poisson_best_reg_param = poisson_best_model.getRegParam()

# Display best parameters
print(f"Best Poisson Regression parameters:")
print(f"Best regParam: {poisson_best_reg_param}")

# Extract metrics for validation curve, both rounded to 4 decimal places
poisson_avg_metrics = [round(float(x), 4) for x in poisson_cv_model.avgMetrics]
poisson_std_metrics = [round(float(x), 4) for x in poisson_cv_model.stdMetrics]

print('\n============The metrics of the Poisson Regression Model=========')
print(f"Avg metrics for Poisson Regression: {poisson_avg_metrics}")
print(f"Standard deviation of metrics for Poisson Regression: {poisson_std_metrics}")

# Function to plot the validation curve for Poisson regression
# Takes in the regularization parameters, average metrics, standard deviation metrics, and the index of the best parameter
def plot_poisson(reg_params, avg_metrics, std_metrics, best_param_index):
    plt.figure(figsize=(12, 8))
    
    # Main plot with error bars
    plt.errorbar(reg_params, avg_metrics, yerr=std_metrics, fmt='-o', 
                 linewidth=2, markersize=8, capsize=5, color='#1f77b4', 
                 label='Poisson Regression')
    
    # Highlight best parameter
    plt.scatter([reg_params[best_param_index]], [avg_metrics[best_param_index]], 
                s=150, c='red', marker='*', 
                label=f'Best regParam: {reg_params[best_param_index]}')
    
    # Add annotation for best performance
    plt.annotate(f'Best RMSE: {avg_metrics[best_param_index]:.4f}',
                 xy=(reg_params[best_param_index], avg_metrics[best_param_index]),
                 xytext=(reg_params[best_param_index]*2, avg_metrics[best_param_index]-0.01),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=12)
    
    # Improve formatting
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Regularization Parameter (λ)', fontsize=14)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=14)
    plt.title('Poisson Regression Validation Curve', fontsize=16)
    
    # Add horizontal line at minimum RMSE
    plt.axhline(y=min(avg_metrics), color='gray', linestyle='--', alpha=0.7)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('./Figures/poisson_validation_curve.png', dpi=300)
    plt.close()

# Plot validation curve for Poisson regression
best_poisson_idx = poisson_avg_metrics.index(min(poisson_avg_metrics))
plot_poisson(reg_params, poisson_avg_metrics, poisson_std_metrics, best_poisson_idx)


# Train a Logistic Regression model for readmission prediction
print("\n================Training Logistic Regression with L2 Regularisation model starts==================")
lr_l2 = LogisticRegression(
featuresCol="medication_features_vector",
labelCol="readmitted",
maxIter=100,
family="binomial",
elasticNetParam=0.0
)

# Create parameter grid for cross-validation of logistic regression with L2 regularization
lr_l2_param_grid = ParamGridBuilder() \
.addGrid(lr_l2.regParam, reg_params) \
.build()

# Create evaluator for logistic regression
lr_evaluator = MulticlassClassificationEvaluator(
labelCol="readmitted", 
predictionCol="prediction",
metricName="accuracy"  
)

# Set up cross-validator with 3 folds
lr_l2_cv = CrossValidator(
estimator=lr_l2,
estimatorParamMaps=lr_l2_param_grid,
evaluator=lr_evaluator,
numFolds=3,
seed=31891,
parallelism=4
)

# Run cross-validation
print("Running Logistic Regression cross-validation....")
lr_l2_cv_model = lr_l2_cv.fit(train_df_transformed)

# Get best parameters and model
best_lr_l2_model = lr_l2_cv_model.bestModel
best_lr_l2_reg_param = best_lr_l2_model.getRegParam()

print(f"Best Logistic Regression with L2 Regularisation parameters:")
print(f"Best regParam: {best_lr_l2_reg_param}")

# Extract metrics for validation curve, both rounded to 4 decimal places
lr_l2_avg_metrics = [round(float(x), 4) for x in lr_l2_cv_model.avgMetrics]
lr_l2_std_metrics = [round(float(x), 4) for x in lr_l2_cv_model.stdMetrics]

print('================== Metrics of Logistic Regression with L2 Regularisation')
print(f"Mean metrics of Logistic Regression with L2 Regularisation: {lr_l2_avg_metrics}")
print(f"Standard deviation of metrics of Logistic Regression with L2 Regularisation: {lr_l2_std_metrics}")

# Function to plot the validation curve for Logistic regression with L2 regularization
# Takes in the regularization parameters, average metrics, standard deviation metrics, and the index of the best parameter
def plot_lr_l2(reg_params, avg_metrics, std_metrics, best_param_index):
    plt.figure(figsize=(12, 8))
    
    # Main plot with error bars
    plt.errorbar(reg_params, avg_metrics, yerr=std_metrics, fmt='-o', 
                 linewidth=2, markersize=8, capsize=5, color='#2ca02c', 
                 label='Logistic Regression with L2')
    
    # Highlight best parameter
    plt.scatter([reg_params[best_param_index]], [avg_metrics[best_param_index]], 
                s=150, c='red', marker='*', 
                label=f'Best regParam: {reg_params[best_param_index]}')
    
    # Add annotation
    plt.annotate(f'Best Accuracy: {avg_metrics[best_param_index]:.4f}',
                 xy=(reg_params[best_param_index], avg_metrics[best_param_index]),
                 xytext=(reg_params[best_param_index]*5, avg_metrics[best_param_index]-0.003),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=12)
    
    # Improve formatting
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Regularization Parameter (λ)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Logistic Regression with L2 Regularization Validation Curve', fontsize=16)
    
    # Add horizontal line at maximum accuracy
    plt.axhline(y=max(avg_metrics), color='gray', linestyle='--', alpha=0.7)
    
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig('./Figures/lr_l2_validation_curve.png', dpi=300)
    plt.close()

# Plot validation curve for Logistic regression with L2 regularization
best_lr_l2_idx = lr_l2_avg_metrics.index(max(lr_l2_avg_metrics))
plot_lr_l2(reg_params, lr_l2_avg_metrics, lr_l2_std_metrics, best_lr_l2_idx)

# ======================== Elastic Net Logistic Regression =======================
# Train a Elastic Net Logistic Regression model 
print("\n=================Training Logistic Regression model with Elastic Net==================")
lr_EN = LogisticRegression(
featuresCol="medication_features_vector",
labelCol="readmitted",
maxIter=100,
family="binomial",
)

# Create parameter grid for cross-validation for elastic net logistic regression
lr_EN_param_grid = ParamGridBuilder() \
    .addGrid(lr_EN.regParam, reg_params) \
    .addGrid(lr_EN.elasticNetParam, elastic_net_params) \
    .build()

# Create evaluator for Elastic Net logistic regression with 3 folds
lr_EN_cv = CrossValidator(
    estimator=lr_EN,
    estimatorParamMaps=lr_EN_param_grid,
    evaluator=lr_evaluator,
    numFolds=3,
    seed=31891,
    parallelism=4
)

# Run cross-validation
lr_EN_cv_model = lr_EN_cv.fit(train_df_transformed)

# Get best parameters and model
best_lr_EN_model = lr_EN_cv_model.bestModel
best_lr_EN_reg_param = best_lr_EN_model.getRegParam()
best_lr_EN_elastic_net_param = best_lr_EN_model.getElasticNetParam()

print(f"Best Logistic Regression with Elastic Net parameters:")
print(f"Best regParam: {best_lr_EN_reg_param}")
print(f"Best elasticNetParam: {best_lr_EN_elastic_net_param}")

# Extract metrics for validation curve, both rounded to 4 decimal places
lr_EN_avg_metrics = [round(float(x), 4) for x in lr_EN_cv_model.avgMetrics]
lr_EN_std_metrics = [round(float(x), 4) for x in lr_EN_cv_model.stdMetrics]

print(f"Average metrics: {lr_EN_avg_metrics}")
print(f"Standard deviation of metrics: {lr_EN_std_metrics}")

# Reshape them into a 2D grid: regParam x elasticNetParam for plotting of elastic net
n_reg_params = len(reg_params)
n_elastic_net_params = len(elastic_net_params)

avg_metrics_reshaped = np.array(lr_EN_avg_metrics).reshape(n_reg_params, n_elastic_net_params)
std_metrics_reshaped = np.array(lr_EN_std_metrics).reshape(n_reg_params, n_elastic_net_params)

# Function to plot the validation curve for Elastic Net logistic regression
# Takes in the regularization parameters, elastic net parameters, average metrics, standard deviation metrics, 
# reshaped metrices and the indices of the best parameters
def plot_elastic_net(reg_params, elastic_net_params, avg_metrics_reshaped, std_metrics_reshaped, best_reg_index, best_elastic_index):
    plt.figure(figsize=(12, 9))
    
    # Custom color palette
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    markers = ['o', 's', 'D', '^', 'v']
    
    # Plot each elastic net parameter as a separate line
    for i, alpha in enumerate(elastic_net_params):
        # Extract metrics for this alpha across all regParam values
        y_values = avg_metrics_reshaped[:, i]
        err_values = std_metrics_reshaped[:, i]
        
        plt.errorbar(x=reg_params,
                    y=y_values,
                    yerr=err_values,
                    fmt=f'-{markers[i]}',
                    linewidth=2, 
                    markersize=8,
                    capsize=5,
                    color=colors[i],
                    label=f'α={alpha}')
    
    # Highlight best parameter combination
    best_reg_param = reg_params[best_reg_index]
    best_elastic_param = elastic_net_params[best_elastic_index]
    best_accuracy = avg_metrics_reshaped[best_reg_index, best_elastic_index]
    
    plt.scatter([best_reg_param], [best_accuracy], 
                s=200, c='red', marker='*', 
                label=f'Best: λ={best_reg_param}, α={best_elastic_param}')
    
    # Add annotation for best performance
    plt.annotate(f'Best Accuracy: {best_accuracy:.4f}',
                xy=(best_reg_param, best_accuracy),
                xytext=(best_reg_param*3, best_accuracy-0.005),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    # Improve formatting
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Regularization Parameter', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Elastic-Net Logistic Regression Validation Curve', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.savefig('./Figures/lr_en_validation_curve.png', dpi=300)
    plt.close()

# Find best combination for Elastic Net Logistic Regression
max_value = np.max(avg_metrics_reshaped)
best_indices = np.where(avg_metrics_reshaped == max_value)
best_reg_idx, best_elastic_idx = best_indices[0][0], best_indices[1][0]
plot_elastic_net(reg_params, elastic_net_params, avg_metrics_reshaped, std_metrics_reshaped, 
                         best_reg_idx, best_elastic_idx)

# Evaluate the best model on the test set
poisson_predictions = poisson_best_model.transform(test_df_transformed)
lr_l2_predictions = best_lr_l2_model.transform(test_df_transformed)
lr_EN_predictions = best_lr_EN_model.transform(test_df_transformed)

poisson_accuracy = poisson_evaluator.evaluate(poisson_predictions)
lr_l2_accuracy = lr_evaluator.evaluate(lr_l2_predictions)
lr_EN_accuracy = lr_evaluator.evaluate(lr_EN_predictions)

# Print the results on the test set of all of the best models to 4 decimal places
print("\nResults on test set of all of the best models:")
print(f"Poisson Regression RMSE: {poisson_accuracy:.4f}")
print(f"Logistic Regression with L2 Regularisation accuracy: {lr_l2_accuracy:.4f}")
print(f"Logistic Regression with Elastic Net accuracy: {lr_EN_accuracy:.4f}")

end = time()

# Stop the Spark session
spark.stop()