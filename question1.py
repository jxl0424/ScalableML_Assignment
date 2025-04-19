from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignment Question 1") \
        .config("spark.local.dir","/mnt/parscratch/users/acp24lj") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz").cache() 
logFile.show(20, False)

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
# data.show(20,False)

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
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(counts)/20,
             f'{int(height):,}',
             ha='center', va='bottom', fontweight='bold')

plt.title('ACADEMIC INSTITUTION REQUESTS BY COUNTRY')
plt.xlabel('Country')
plt.ylabel('Number of Requests')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('academic_requests_by_country.png')
plt.show()             

# QUESTION B
# ===============================================================================================
def extract_institutions(df, suffix):
    # Extract just institution name (without TLD)
    if suffix == '.edu':
        # For US, extract everything before .edu
        institution_df = df.withColumn('institution', regexp_extract(col('host'), r'([^.]+)\.edu', 1))
    elif suffix == '.ac.uk':
        # For UK, extract everything before .ac.uk
        institution_df = df.withColumn('institution', regexp_extract(col('host'), r'([^.]+)\.ac\.uk', 1))
    else:  # .edu.au
        # For Australia, extract everything before .edu.au
        institution_df = df.withColumn('institution', regexp_extract(col('host'), r'([^.]+)\.edu\.au', 1))
    
    # Count unique institutions
    unique_count = institution_df.select('institution').distinct().count()
    
    # Get top 9 institutions by request count
    top_institutions = institution_df.groupBy('institution') \
        .count() \
        .orderBy(desc('count')) \
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