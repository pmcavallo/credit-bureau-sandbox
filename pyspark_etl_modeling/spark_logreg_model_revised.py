
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.stat import KolmogorovSmirnovTest

# Initialize Spark session
spark = SparkSession.builder.appName("CreditRiskLogRegModel").getOrCreate()

# Load the cleaned and engineered data from Parquet
df = spark.read.parquet("output/credit_data_cleaned2.parquet")

# Optional: Convert categorical feature to numeric
df = df.withColumn("util_bin_indexed", 
                   StringIndexer(inputCol="util_bin", outputCol="util_bin_indexed").fit(df).transform(df)["util_bin_indexed"])

# Assemble feature vector
assembler = VectorAssembler(
    inputCols=[
        "fico_score", "loan_amount", "cltv", "util_bin_indexed", 
        "delinq_flag", "high_risk_flag"
    ],
    outputCol="features"
)
assembled = assembler.transform(df)

# Train-test split
train, test = assembled.randomSplit([0.7, 0.3], seed=42)

# Fit logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", probabilityCol="probability", predictionCol="prediction")
model = lr.fit(train)

# Evaluate model
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

print(f"✅ Model AUC: {auc:.4f}")

# Show some predictions
predictions.select("fico_score", "loan_amount", "probability", "prediction").show(5)

# KS Test (optional)
ks_result = KolmogorovSmirnovTest.test(predictions, "probability", "loan_status_flag").head()
print(f"✅ KS Statistic: {ks_result.statistic:.4f} | p-value: {ks_result.pValue:.4f}")

spark.stop()
