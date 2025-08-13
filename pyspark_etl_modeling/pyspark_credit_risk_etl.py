print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import *

# Initialize Spark Session
spark = SparkSession.builder     .appName("Credit Risk PySpark ETL")     .getOrCreate()

# Define schema explicitly
schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("fico_score", IntegerType(), True),
    StructField("loan_amount", DoubleType(), True),
    StructField("tenure_months", IntegerType(), True),
    StructField("state", StringType(), True),
    StructField("plan_type", StringType(), True),
    StructField("monthly_income", DoubleType(), True),
    StructField("date_issued", StringType(), True),  # Optional: convert to DateType later
    StructField("loan_status", StringType(), True),
    StructField("loan_status_flag", IntegerType(), True),
    StructField("credit_utilization", DoubleType(), True),
    StructField("has_bankruptcy", IntegerType(), True)
])

# Load CSV file
df = spark.read.csv("C:/credit_risk_project/data/raw/credit_data_aws_flagship.csv", schema=schema, header=True)

# Feature Engineering
df = df.withColumn("cltv", (col("monthly_income") * col("tenure_months") / 12) - col("loan_amount"))

df = df.withColumn("util_bin", when(col("credit_utilization") < 0.3, "Low")
                                   .when(col("credit_utilization") < 0.6, "Medium")
                                   .otherwise("High"))

df = df.withColumn("delinq_flag", (col("loan_status_flag") == 1).cast("int"))

df = df.withColumn("high_risk_flag", when((col("fico_score") < 580) |
                                          (col("plan_type") == "Business") |
                                          (col("has_bankruptcy") == 1), 1).otherwise(0))

# Show sample of engineered features
df.select("customer_id", "fico_score", "loan_amount", "cltv", "util_bin", "delinq_flag", "high_risk_flag").show(10)

# Drop rows with missing target or core variables (optional cleanup)
df_cleaned = df.dropna(subset=["fico_score", "loan_amount", "monthly_income", "loan_status_flag"])

# Write cleaned DataFrame to Parquet
df_cleaned.write.mode("overwrite").parquet("output/credit_data_cleaned.parquet")

# Export a 100-row sample to CSV for Pandas EDA
sample_df = df_cleaned.limit(100).toPandas()
sample_df.to_csv("sample_for_pandas_eda.csv", index=False)

# Stop Spark session
spark.stop()
