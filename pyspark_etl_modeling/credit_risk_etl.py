
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# 1. Start a Spark session
spark = SparkSession.builder \
    .appName("Credit Risk ETL") \
    .getOrCreate()

# 2. Load raw CSV data
input_path = "C:/credit_risk_project/data/raw/credit_data_aws_flagship.csv"
df = spark.read.csv(input_path, header=True, inferSchema=True)

print("✅ Raw data loaded")
df.printSchema()

# 3. Clean and transform
df_clean = df.withColumn(
    "fico_score",
    when(col("fico_score").isNull(), 600).otherwise(col("fico_score"))
).withColumn(
    "loan_status_flag",
    when(col("loan_status") == "Default", 1).otherwise(0)
)

df_clean = df_clean.dropna()

print("✅ Data cleaned")
df_clean.show(5)

# 4. Save cleaned data
output_path = "../data/processed_spark/credit_data_clean.parquet"
df_clean.write.mode("overwrite").parquet(output_path)

print(f"✅ Cleaned data written to {{output_path}}")

# 5. Stop Spark
spark.stop()
