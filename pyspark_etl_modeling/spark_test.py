from pyspark.sql import SparkSession

print("🚀 Spark test starting...")

spark = SparkSession.builder \
    .appName("MinimalSparkSession") \
    .master("local[*]") \
    .getOrCreate()

print("✅ SparkSession started successfully!")

spark.stop()







