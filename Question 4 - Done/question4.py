import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.sql.functions import col, rand, avg, count, lit
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.types import FloatType, IntegerType
from time import time

start = time()
# Initialize Spark session
spark = SparkSession.builder \
        .appName("Question 4") \
        .config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
        .config("spark.executor.javaOptions", "-Xss32m") \
        .config("spark.driver.javaOptions", "-Xss32m") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

file_path = "./Data/ml-25m"
# Load the ratings data
ratings = spark.read.csv(f"{file_path}/ratings.csv", header=True, inferSchema=True)
ratings = ratings.select(col("userId").cast("integer"), 
                         col("movieId").cast("integer"), 
                         col("rating").cast("float"))

# Load movies data
movies = spark.read.csv(f"{file_path}/movies.csv", header=True, inferSchema=True)

# Load genome scores and tags (for potential additional analysis)
genome_scores = spark.read.csv(f"{file_path}/genome-scores.csv", header=True, inferSchema=True)
genome_tags = spark.read.csv(f"{file_path}/genome-tags.csv", header=True, inferSchema=True)

# Select only the columns we need and cache the data
genome_scores.cache()
genome_tags.cache()
ratings.cache()
ratings.show(20, False)
print(f"Total number of ratings: {ratings.count()}")

# Add a fold column for cross-validation (4 folds)
# We'll use random partitioning with consistent seed for reproducibility
ratings_with_folds = ratings.withColumn("fold", (rand(seed=31891) * 4).cast("int"))
ratings_with_folds.show(20, False)
ratings_with_folds.groupBy("fold").count().show()
ratings_with_folds.cache()

als_params =[
    {
    # Default parameters
    "userCol": "userId",
    "itemCol": "movieId",
    "coldStartStrategy": "drop", 
    "seed" : 31891
    },
    {
    "maxIter": 20,
    "rank": 20,
    "regParam": 0.1,
    "userCol": "userId",
    "itemCol": "movieId",
    "coldStartStrategy": "drop", 
    "seed" : 31891
    },
    {
    "maxIter": 20,
    "rank": 50,
    "regParam": 0.001,
    "userCol": "userId",
    "itemCol": "movieId",
    "coldStartStrategy": "drop",
    "seed" : 31891
    }
]

# Initialize RMSE and MAE evaluators
rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

# Store results
results = {f"model{i+1}": {"rmse": [], "mae": []} for i in range(3)}

# Perform 4-fold cross-validation
for fold in range(4):
    print(f"\nProcessing fold {fold + 1}/4")
    
    # For each fold split into train and test
    train = ratings_with_folds.filter(col("fold") != fold)
    test = ratings_with_folds.filter(col("fold") == fold)

    for i, params in enumerate(als_params):
        print(f"Training ALS model {i+1} for fold {fold+1}")
        als = ALS(**params)
        model = als.fit(train)
        predictions = model.transform(test)
        predictions = predictions.filter(col("prediction").isNotNull())

        rmse = rmse_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)
        results[f"model{i+1}"]["rmse"].append(rmse)
        results[f"model{i+1}"]["mae"].append(mae)
        print(f"Model {i+1} Fold {fold+1} - RMSE: {rmse}, MAE: {mae}")

# Compute mean and std for RMSE and MAE across folds
results_table = {}
for model in ['model1', 'model2', 'model3']:
    rmse_values = results[model]["rmse"]
    mae_values = results[model]["mae"]
    results_table[model] = {
        "RMSE Fold 1": rmse_values[0],
        "RMSE Fold 2": rmse_values[1],
        "RMSE Fold 3": rmse_values[2],
        "RMSE Fold 4": rmse_values[3],
        "RMSE Mean": np.mean(rmse_values),
        "RMSE Std": np.std(rmse_values),
        "MAE Fold 1": mae_values[0],
        "MAE Fold 2": mae_values[1],
        "MAE Fold 3": mae_values[2],
        "MAE Fold 4": mae_values[3],
        "MAE Mean": np.mean(mae_values),
        "MAE Std": np.std(mae_values)
    }    

# Create table for results
table_df = pd.DataFrame(results_table).T
table_df = table_df[[
    "RMSE Fold 1", "RMSE Fold 2", "RMSE Fold 3", "RMSE Fold 4", "RMSE Mean", "RMSE Std",
    "MAE Fold 1", "MAE Fold 2", "MAE Fold 3", "MAE Fold 4", "MAE Mean", "MAE Std"
]]
print("\nMetrics Table:")
print(table_df)

# Visualization
models = ["Model 1", "Model 2", "Model 3"]
rmse_means = [results_table[model]["RMSE Mean"] for model in ["model1", "model2", "model3"]]
rmse_stds = [results_table[model]["RMSE Std"] for model in ["model1", "model2", "model3"]]
mae_means = [results_table[model]["MAE Mean"] for model in ["model1", "model2", "model3"]]
mae_stds = [results_table[model]["MAE Std"] for model in ["model1", "model2", "model3"]]

# RMSE Plot
plt.figure(figsize=(8, 6))
x = np.arange(len(models))
bars = plt.bar(x, rmse_means, yerr=rmse_stds, capsize=5, color="skyblue", edgecolor="black")
plt.xticks(x, models)
plt.ylabel("RMSE")
plt.title("Mean RMSE with Standard Deviation for ALS Models")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
             ha='center', va='bottom', fontsize=10)

plt.savefig("./Figures/rmse_comparison.png")
plt.close()

# MAE Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(x, mae_means, yerr=mae_stds, capsize=5, color="lightcoral", edgecolor="black")
plt.xticks(x, models)
plt.ylabel("MAE")
plt.title("Mean MAE with Standard Deviation for ALS Models")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
             ha='center', va='bottom', fontsize=10)

plt.savefig("./Figures/mae_comparison.png")
plt.close()

# =================================== PART B =================================
k = 19
def train_get_factors(train):
    als = ALS(userCol = "userId", itemCol = "movieId", seed = 31891, coldStartStrategy = "drop")
    model = als.fit(train)
    return model.itemFactors

def item_factors_clustering(item_factors, k, seed):
    kmeans = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(item_factors)
    predictions = model.transform(item_factors)
    return predictions.select("id", "cluster")

def get_top_clusters(clustered_factors):
    cluster_sizes = clustered_factors.groupBy("cluster").count().orderBy(col("count").desc())
    top_clusters = cluster_sizes.limit(3).select("cluster").collect()
    return [row["cluster"] for row in top_clusters], cluster_sizes    

# Function to get top 3 tags for a cluster
def get_top_tags_for_cluster(cluster_id, clustered_factors, genome_scores, genome_tags):
    # Get movies in the cluster
    cluster_movies = clustered_factors.filter(col("cluster") == cluster_id).select("id").withColumnRenamed("id", "movieId")
    
    # Join with genome scores to get relevance scores
    cluster_scores = cluster_movies.join(genome_scores, "movieId")
    
    # Compute average relevance per tag
    tag_scores = cluster_scores.groupBy("tagId").agg(avg("relevance").alias("avg_relevance"))
    
    # Get top 3 tags by average relevance
    top_tags = tag_scores.orderBy(col("avg_relevance").desc()).limit(3)
    
    # Join with genome_tags to get tag names
    top_tags_with_names = top_tags.join(genome_tags, "tagId").select("tagId", "tag", "avg_relevance")
    
    return top_tags_with_names

# Function to count movies with a tag in a cluster
def count_movies_with_tag(cluster_id, tag_id, clustered_factors, genome_scores, relevance_threshold=0.5):
    # Get movies in the cluster
    cluster_movies = clustered_factors.filter(col("cluster") == cluster_id).select("id").withColumnRenamed("id", "movieId")
    
    # Join with genome scores for the specific tag
    tag_scores = genome_scores.filter(col("tagId") == tag_id).filter(col("relevance") > relevance_threshold)
    
    # Count movies in the cluster with the tag
    movies_with_tag = cluster_movies.join(tag_scores, "movieId").count()
    
    return movies_with_tag

for fold in range(4):
    print(f"\nProcessing Fold {fold + 1}/4")
    
    # Split into train and test
    train = ratings_with_folds.filter(col("fold") != fold)
    test = ratings_with_folds.filter(col("fold") == fold)
    
    # Train ALS and get item factors
    item_factors = train_get_factors(train)
    
    # Cluster item factors with K-means
    clustered_factors = item_factors_clustering(item_factors, k, seed = 31891)
    
    # Get top 3 clusters
    top_cluster_ids, cluster_sizes = get_top_clusters(clustered_factors)
    print(f"Top 3 clusters (by size) in Fold {fold + 1}:")
    cluster_sizes_df = cluster_sizes.filter(col("cluster").isin(top_cluster_ids)).orderBy(col("count").desc())
    cluster_sizes_df.show(truncate=False)
    
    # Process each top cluster
    for cluster_id in top_cluster_ids:
        print(f"\nCluster {cluster_id} in Fold {fold + 1}:")
        
        # Get top 3 tags
        top_tags_df = get_top_tags_for_cluster(cluster_id, clustered_factors, genome_scores, genome_tags)
        print(f"Top 3 tags for Cluster {cluster_id}:")
        top_tags_df.show(truncate=False)
        
        # Count movies with each tag
        print(f"Number of movies with each tag in Cluster {cluster_id}:")
        for row in top_tags_df.collect():
            tag_id = row["tagId"]
            tag_name = row["tag"]
            count = count_movies_with_tag(cluster_id, tag_id, clustered_factors, genome_scores)
            print(f"Tag '{tag_name}' (tagId={tag_id}): {count} movies")

# Stop Spark session
spark.stop()