from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from time import time

start = time()

spark = SparkSession.builder \
        .appName("Question 1") \
        .config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") 
print("=================Starting Question 1====================")
# Step 1 - Read log file from Data folder
logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz")

# Step 2 - Using RegEX to extract relevant fields and create new columns for the DataFrame
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
            .withColumn('timestamp', F.regexp_extract('value', '.* - - \\[(.*?)\\].*',1)) \
            .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
            .withColumn('HTTP reply code', F.element_at(F.split('value', ' '), -2).cast("int")) \
            .withColumn('bytes in the reply', F.element_at(F.split('value', ' '), -1).cast("int")) \
            .withColumn('day', F.regexp_extract('timestamp', r'(\d+)/\w+/\d+', 1).cast('int')) \
            .withColumn('hour', F.regexp_extract('timestamp', r'\d+/\w+/\d+:(\d+):', 1).cast('int')) \
            .drop("value") \
            .cache()

# QUESTION A - Count requests from academic institutions
# ===============================================================================================
# Step 3 - Create a country code column and filter by the suffixes for academic institutions
# Since we are only interested in US, UK, and AUS institutions, filter other countries to exclude them
country_data = data.withColumn(
    "country_code", 
    F.when(F.col("host").endswith(".edu"), "US")
     .when(F.col("host").endswith(".ac.uk"), "UK")
     .when(F.col("host").endswith(".edu.au"), "AUS")
).filter(F.col("country_code").isin("US", "UK", "AUS"))

# Get counts from country code
country_counts = country_data.groupBy("country_code").count().collect()

# Extract counts for US, UK, and AUS 
country_count_dict = {row["country_code"]: row["count"] for row in country_counts}
us_request = country_count_dict.get("US", 0)
uk_request = country_count_dict.get("UK", 0)
aus_request = country_count_dict.get("AUS", 0)

# Display total requests from each country
print("\nTotal requests from academic institutions are as follows:")
print(f"The total number of requests from US academic institution is: {us_request}")
print(f"The total number of requests from UK academic institution is: {uk_request}")
print(f"The total number of requests from AUS academic institution is: {aus_request}")

# Visualization of results
countries = ['USA', 'UK', 'AUS']
requests = [us_request, uk_request, aus_request]
plt.figure(figsize=(10, 6))
bars = plt.bar(countries, requests, color=['blue', 'red', 'green'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(requests)/20,
             f'{int(height):,}',
             ha='center', va='bottom', fontweight='bold')

plt.title('Academic Institution Requests by US, UK, and AUS')
plt.xlabel('Country')
plt.ylabel('Number of Requests')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./Figures/academic_requests_by_country.png')
plt.close()            

# QUESTION B - Identifying unique institutions and frequent hosts
# ===============================================================================================
# Step 1 - Extract institution names for all countries based on their hostnames (prefixes)
# Store the institution names in a new dataframe with new column name "institution"
institution_data = country_data.withColumn(
    "institution",
    F.when(F.col("country_code") == "US", F.regexp_extract(F.col("host"), r"([^.]+)\.edu", 1))
     .when(F.col("country_code") == "UK", F.regexp_extract(F.col("host"), r"([^.]+)\.ac\.uk", 1))
     .when(F.col("country_code") == "AUS", F.regexp_extract(F.col("host"), r"([^.]+)\.edu\.au", 1))
)

# Step 2 - Calculate unique institution counts by country
unique_institutions = institution_data.select("country_code", "institution").distinct() \
    .groupBy("country_code").count().collect()
unique_count_dict = {row["country_code"]: row["count"] for row in unique_institutions}

# Step 3 - Get top 9 institutions for each country 
# Group the data first
institution_counts = institution_data.filter(F.col("institution").isNotNull()) \
    .groupBy("country_code", "institution").count()

# For each country code, get top 9 institutions separately and union results
country_codes = [row["country_code"] for row in institution_counts.select("country_code").distinct().collect()]
top_institutions_df = None

for country_code in country_codes:
    country_top9 = institution_counts.filter(F.col("country_code") == country_code) \
        .orderBy(F.desc("count")) \
        .limit(9)
    
    if top_institutions_df is None:
        top_institutions_df = country_top9
    else:
        top_institutions_df = top_institutions_df.union(country_top9)

# Step 4 - Collect the results
top_institutions = top_institutions_df.orderBy("country_code", F.desc("count")).collect()

# Organize top institutions by country for reporting
top_by_country = {}
for row in top_institutions:
    country = row["country_code"]
    if country not in top_by_country:
        top_by_country[country] = []
    top_by_country[country].append({"institution": row["institution"], "count": row["count"]})

# Step 5 - Display results
print("\n==============Identifying top 9 unique academic institutions from each country======================")
print(f"US unique academic institutions: {unique_count_dict.get('US', 0)}")
print("Top 9 US institutions:")
for inst in top_by_country.get('US', []):
    print(f"  {inst['institution']}: {inst['count']} requests")

print(f"\nUK unique academic institutions: {unique_count_dict.get('UK', 0)}")
print("Top 9 UK institutions:")
for inst in top_by_country.get('UK', []):
    print(f"  {inst['institution']}: {inst['count']} requests")

print(f"\nAustralia unique academic institutions: {unique_count_dict.get('AUS', 0)}")
print("Top 9 Australian institutions:")
for inst in top_by_country.get('AUS', []):
    print(f"  {inst['institution']}: {inst['count']} requests")

# Get Sheffield's rank
# First get all UK institution counts and sort them
uk_institution_counts = institution_data.filter(F.col("country_code") == "UK") \
    .groupBy("institution").count() \
    .orderBy(F.desc("count"))

# Convert to pandas to calculate rank (more efficient for small result sets)
uk_institutions_pd = uk_institution_counts.toPandas()

# Find Sheffield's rank 
# Implemented some safety precautions in case Sheffield is not found in the dataframe
sheffield_rank = None
if not uk_institutions_pd.empty:
    # Find the position of Sheffield in the sorted dataframe
    sheffield_rows = uk_institutions_pd[uk_institutions_pd['institution'] == 'shef']
    if not sheffield_rows.empty:
        # Get Sheffield's position (0-based index) and add 1 for rank
        sheffield_index = uk_institutions_pd.index[uk_institutions_pd['institution'] == 'shef'][0]
        sheffield_rank = {"rank": sheffield_index + 1}

if sheffield_rank:
    print(f"\nSheffield's rank in the UK academic institutions: {sheffield_rank['rank']}")
else:
    print("\nSheffield not found in UK academic institutions")

# Question C - Distribution Visualization
# ===============================================================================================
# Function to create distribution visualization for a given country
# The function takes the country name, country code, and top 9 institutions list as input
# and generates a pie chart showing the distribution of requests from academic institutions in that country.
# instiution other than top 9 are grouped into "Others"

def create_distribution_visualization(country_name, country_code, top9_list):
    # Calculate total count for the country
    country_total = country_count_dict.get(country_code, 0)
    
    # Get institution names and counts from top9 list
    top9_institutions = [item["institution"] for item in top9_list]
    top9_counts = [item["count"] for item in top9_list]
    
    # Calculate count for "Others" - total count minus sum of top 9
    top9_count_sum = sum(top9_counts)
    others_count = country_total - top9_count_sum
    
    # Create data for plotting
    labels = top9_institutions + ["Others"]
    sizes = top9_counts + [others_count]
    
    # Plot pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, 
            shadow=True, explode=[0.05] * len(labels))
    
    # Add legend with counts
    legend_labels = [f"{inst} ({count})" for inst, count in zip(top9_institutions, top9_counts)]
    legend_labels.append(f"Others ({others_count})")
    plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(f'Distribution of Requests from {country_name} Academic Institutions')
    plt.tight_layout()
    plt.savefig(f'./Figures/{country_name.lower().replace(" ", "_")}_distribution.png')
    plt.close()

# Create distribution visualizations for US, UK, and AUS
if 'US' in top_by_country:
    create_distribution_visualization("United States", "US", top_by_country['US'])
if 'UK' in top_by_country:
    create_distribution_visualization("United Kingdom", "UK", top_by_country['UK'])
if 'AUS' in top_by_country:
    create_distribution_visualization("Australia", "AUS", top_by_country['AUS'])

# QUESTION D - Heatmap Visualization and Analysis
# ===============================================================================================
# Function to generate heatmap for a given country and institution
# The function takes the country name, country code, and top institution as input
def create_heatmap(country_name, country_code, top_institution):
    # Get the top institution name
    inst_name = top_institution["institution"]
    
    # Pre-aggregate the data in Spark
    heatmap_data = institution_data.filter(
        (F.col("country_code") == country_code) & 
        (F.col("institution") == inst_name)
    ).groupBy("day", "hour").count().collect()
    
    # Create a 24x31 array (hours x days) filled with zeros
    heat_array = np.zeros((24, 31))
    
    # Fill the array with counts
    for row in heatmap_data:
        day_idx = row['day'] - 1  # Convert to 0-based index
        hour_idx = row['hour']
        if 0 <= day_idx < 31 and 0 <= hour_idx < 24:  # Validate indices
            heat_array[hour_idx, day_idx] = row['count']
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heat_array, cmap="YlOrRd", 
                xticklabels=range(1, 32),  # Days of July
                yticklabels=range(24),     # Hours of day
                cbar_kws={'label': 'Number of Requests'})
    
    plt.title(f'Access Pattern for {inst_name} ({country_name})')
    plt.xlabel('Day of Month (July 1995)')
    plt.ylabel('Hour of Day (0-23)')
    plt.tight_layout()
    plt.savefig(f'./Figures/{country_name.lower().replace(" ", "_")}_{inst_name}_heatmap.png')
    plt.close()

# Create heatmaps for top institution in each country
if 'US' in top_by_country and top_by_country['US']:
    create_heatmap("United States", "US", top_by_country['US'][0])
if 'UK' in top_by_country and top_by_country['UK']:
    create_heatmap("United Kingdom", "UK", top_by_country['UK'][0])
if 'AUS' in top_by_country and top_by_country['AUS']:
    create_heatmap("Australia", "AUS", top_by_country['AUS'][0])

end = time()
# Stop the Spark session
spark.stop()