
from pyspark.sql import SparkSession

print("🚀 Spark check starting...")

try:
    spark = SparkSession.builder \
        .appName("SparkCheck") \
        .getOrCreate()

    print("✅ SparkSession started successfully!")

    spark.stop()
    print("🛑 SparkSession stopped.")
except Exception as e:
    print("❌ SparkSession failed to start:")
    print(e)
