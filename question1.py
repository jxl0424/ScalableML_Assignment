from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignment Question 1") \
        .config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz").cache() 

# split into 5 columns using regex and split
# data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
#                 .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
#                 .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
#                 .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
#                 .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
            .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
            .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
            .withColumn('HTTP reply code', F.element_at(F.split('value', ' '), -2).cast("int")) \
            .withColumn('bytes in the reply', F.element_at(F.split('value', ' '), -1).cast("int")) \
            .drop("value").cache()

data = data.withColumn('day', F.regexp_extract('timestamp', r'(\d+)/\w+/\d+', 1).cast('int')) \
           .withColumn('hour', F.regexp_extract('timestamp', r'\d+/\w+/\d+:(\d+):', 1).cast('int'))

# QUESTION A 
# ===============================================================================================
# Filter for 3 countries
us_data = data.filter(F.col('host').endswith('.edu')).cache() # filter the requests that are from US
uk_data = data.filter(F.col('host').endswith('.ac.uk')).cache() # filter the requests that are from UK
aus_data = data.filter(F.col('host').endswith('.edu.au')).cache() # filter the requests that are from AU

# The count for each country
us_request = us_data.count()
uk_request = uk_data.count()
aus_request = aus_data.count()

# Display results
print(f"The total number of requests from US academic institution is: {us_request}")
print(f"The total number of requests from UK academic institution is: {uk_request}")
print(f"The total number of requests from AU academic institution is: {aus_request}")

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

plt.title('ACADEMIC INSTITUTION REQUESTS BY COUNTRY')
plt.xlabel('Country')
plt.ylabel('Number of Requests')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./Figures/academic_requests_by_country.png')            

# QUESTION B
# ===============================================================================================
def extract_institutions(df, suffix):
    # Extract just institution name (without TLD)
    if suffix == '.edu':
        # For US, extract everything before .edu
        institution_df = df.withColumn('institution', F.regexp_extract(F.col('host'), r'([^.]+)\.edu', 1))
    elif suffix == '.ac.uk':
        # For UK, extract everything before .ac.uk
        institution_df = df.withColumn('institution', F.regexp_extract(F.col('host'), r'([^.]+)\.ac\.uk', 1))
    else:  # .edu.au
        # For Australia, extract everything before .edu.au
        institution_df = df.withColumn('institution', F.regexp_extract(F.col('host'), r'([^.]+)\.edu\.au', 1))
    
    # Count unique institutions
    unique_count = institution_df.select('institution').distinct().count()
    
    # Get top 9 institutions by request count
    top_institutions = institution_df.groupBy('institution') \
        .count() \
        .orderBy(F.desc('count')) \
        .limit(9) \
        .collect()
    
    return unique_count, top_institutions, institution_df

unique_us_requests, top9_us, us_institution_df = extract_institutions(us_data, '.edu')
unique_uk_requests, top9_uk, uk_institution_df = extract_institutions(uk_data, '.ac.uk') 
unique_aus_requests, top9_aus, aus_institution_df = extract_institutions(aus_data, '.edu.au')   

# Display results
print(f"US unique academic institutions: {unique_us_requests}")
print("Top 9 US institutions:")
for inst in top9_us:
    print(f"  {inst['institution']}: {inst['count']} requests")

print(f"\nUK unique academic institutions: {unique_uk_requests}")
print("Top 9 UK institutions:")
for inst in top9_uk:
    print(f"  {inst['institution']}: {inst['count']} requests")

print(f"\nAustralia unique academic institutions: {unique_aus_requests}")
print("Top 9 Australian institutions:")
for inst in top9_aus:
    print(f"  {inst['institution']}: {inst['count']} requests")

uk_institutions = uk_institution_df.groupBy('institution') \
    .count() \
    .orderBy(F.desc('count')) 
                       
uk_institutions_pd = uk_institutions.toPandas()
uk_institutions_pd['rank'] = uk_institutions_pd['count'].rank(method='min', ascending=False)

sheffield_rank = uk_institutions_pd[uk_institutions_pd['institution']=='shef']['rank'].values[0]

print(f"\nSheffield's rank in the UK academic institutions: {sheffield_rank}")

# Question C
# ===============================================================================================
def create_distribution_visualization(country_name, inst_counts_df, top9_list):
    # Convert to pandas for easier manipulation
    inst_counts_pandas = inst_counts_df.groupBy('institution').count().toPandas()
    
    # Get institution names from top9
    top9_institutions = [row["institution"] for row in top9_list]
    
    # Calculate count for "Others" - total count minus sum of top 9
    total_count = inst_counts_pandas['count'].sum()
    top9_count_sum = sum(row["count"] for row in top9_list)
    others_count = total_count - top9_count_sum
    
    # Create data for plotting
    labels = top9_institutions + ["Others"]
    sizes = [row["count"] for row in top9_list] + [others_count]
    
    # Plot pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, 
            shadow=True, explode=[0.05] * len(labels))
    
    # Add legend with counts
    legend_labels = [f"{inst} ({count})" for inst, count in zip(top9_institutions, [row["count"] for row in top9_list])]
    legend_labels.append(f"Others ({others_count})")
    plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(f'Distribution of Requests from {country_name} Academic Institutions')
    plt.tight_layout()
    plt.savefig(f'./Figures/{country_name.lower().replace(" ", "_")}_distribution.png')
    plt.close()

# Create distribution visualizations for each country
create_distribution_visualization("United States", us_institution_df, top9_us)
create_distribution_visualization("United Kingdom", uk_institution_df, top9_uk)
create_distribution_visualization("Australia", aus_institution_df, top9_aus)

# QUESTION D
# ===============================================================================================
# Function to create heatmap using aggregated Spark data
# Function to create heatmap using aggregated Spark data
def create_heatmap(country_name, inst_df, top_institution):
    # Get the top institution name
    inst_name = top_institution["institution"]
    
    # Filter data for just that institution
    top_inst_df = inst_df.filter(F.col("institution") == inst_name)
    
    # Make sure day and hour columns exist in the dataframe
    # If they don't exist in inst_df, we need to join with the original data
    if "day" not in top_inst_df.columns or "hour" not in top_inst_df.columns:
        # Join with original data to get timestamp info
        top_inst_df = top_inst_df.join(data.select("host", "day", "hour"), on="host")
    
    # Aggregate data by day and hour
    heatmap_data = top_inst_df.groupBy("day", "hour").count()
    
    # Collect the data to the driver
    heat_rows = heatmap_data.collect()
    
    # Create a 24x31 array (hours x days) filled with zeros
    heat_array = np.zeros((24, 31))
    
    # Fill the array with counts
    for row in heat_rows:
        day_idx = row['day'] - 1  # Convert to 0-based index
        hour_idx = row['hour']
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

# Create heatmaps for each country
if top9_us:
    create_heatmap("United States", us_institution_df, top9_us[0])
if top9_uk:
    create_heatmap("United Kingdom", uk_institution_df, top9_uk[0])
if top9_aus:
    create_heatmap("Australia", aus_institution_df, top9_aus[0])