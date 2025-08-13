print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder.appName("Credit Risk ETL Updated").getOrCreate()

# Define schema explicitly
schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("fico_score", DoubleType(), True),
    StructField("loan_amount", DoubleType(), True),
    StructField("tenure_months", IntegerType(), True),
    StructField("state", StringType(), True),
    StructField("plan_type", StringType(), True),
    StructField("monthly_income", DoubleType(), True),
    StructField("date_issued", StringType(), True),
    StructField("loan_status", StringType(), True),
    StructField("loan_status_flag", IntegerType(), True),
    StructField("credit_utilization", DoubleType(), True),
    StructField("has_bankruptcy", IntegerType(), True)
])

# Load data
df = spark.read.csv("C:/credit_risk_project/data/raw/credit_data_aws_flagship2.csv", schema=schema, header=True)
print(f"✅ Loaded {df.count()} rows")


# Feature engineering
df = df.withColumn("loan_status_flag", col("loan_status_flag").cast("int"))

df = df.withColumn("cltv", (col("monthly_income") * col("tenure_months") / 12) - col("loan_amount"))

df = df.withColumn("util_bin", when(col("credit_utilization") < 0.3, "Low")
                                   .when(col("credit_utilization") < 0.6, "Medium")
                                   .otherwise("High"))

df = df.withColumn("delinq_flag", (col("loan_status_flag") == 1).cast("int"))

df = df.withColumn("high_risk_flag", when((col("fico_score") < 580) |
                                          (col("plan_type") == "Business") |
                                          (col("has_bankruptcy") == 1), 1).otherwise(0))

# Optional: preview
df.select("customer_id", "fico_score", "loan_amount", "cltv", "util_bin", "delinq_flag", "high_risk_flag").show(10)

# Drop rows with missing target or important fields
df_cleaned = df.dropna(subset=["fico_score", "loan_amount", "monthly_income", "loan_status_flag"])

df.groupBy("loan_status_flag").count().show()
df.select("fico_score").filter("fico_score IS NULL").count()

# ✅ Save full cleaned dataset to new Parquet file
df_cleaned.write.mode("overwrite").parquet("output/credit_data_cleaned2.parquet")

# Stop Spark session
spark.stop()
