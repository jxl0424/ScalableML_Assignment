from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignment Question 2") \
        .config("spark.sql.debug.maxToStringFields", 1000) \
        .config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

df_pandas = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "brandao/diabetes",
    "diabetic_data.csv" 
)
# Preprocess the data
df = spark.createDataFrame(df_pandas)

medication_features = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]
df = df.withColumn("readmitted", 
                  when(col("readmitted") == "NO", 0).otherwise(1))
df = df.withColumn("readmitted", col("readmitted").cast(IntegerType()))

df.show()
# # 2. Select one numeric feature - I'll choose time_in_hospital
# selected_numeric_feature = "time_in_hospital"

# # 3. One-hot encode the medication features
# # First, create string indexers for each medication feature
# indexers = [StringIndexer(inputCol=feature, outputCol=f"{feature}_index")
#            for feature in medication_features]

# # Then, create one-hot encoders for each indexed feature
# encoders = [OneHotEncoder(inputCol=f"{feature}_index", outputCol=f"{feature}_vec")
#            for feature in medication_features]

# # Create a list to store all feature columns for the final vector assembler
# feature_cols = [f"{col_name}_vec" for col_name in medication_features] + [selected_numeric_feature]

# # Create a vector assembler to combine all features into a single vector column
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# # Create and set up the pipeline
# pipeline = Pipeline(stages=indexers + encoders + [assembler])

# # Fit the pipeline to the data
# model = pipeline.fit(df)
# transformed_df = model.transform(df)

# # Keep only the necessary columns
# final_df = transformed_df.select("features", "readmitted")

# final_df.show(5)