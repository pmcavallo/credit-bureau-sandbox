# 🚀 Script has started running...
print("🚀 Script has started running...")
"""
Logistic Regression with Class Weights and Regularization
For Credit Risk Modeling using PySpark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Start Spark session
spark = SparkSession.builder     .appName("CreditRiskLogisticRegression")     .getOrCreate()

print("✅ Spark session started.")

# Load the transformed data (adjust path if needed)
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")
print(f"✅ Loaded transformed data. Row count: {df.count()}")

# STEP 1: Create class weights
label_counts = df.groupBy("loan_status_flag").count().toPandas()
majority = label_counts["count"].max()
weights = {
    int(row["loan_status_flag"]): majority / row["count"]
    for _, row in label_counts.iterrows()
}

@udf(DoubleType())
def get_weight(label):
    return float(weights[int(label)])

df = df.withColumn("classWeightCol", get_weight(df["loan_status_flag"]))
print(f"✅ Added classWeightCol. Weights: {weights}")

# STEP 2: Assemble features
features = ['fico_score', 'loan_amount', 'cltv', 'delinq_flag', 'high_risk_flag']
assembler = VectorAssembler(inputCols=features, outputCol="features")

# STEP 3: Logistic regression with regularization
lr = LogisticRegression(
    labelCol="loan_status_flag",
    featuresCol="features",
    weightCol="classWeightCol",
    regParam=0.1,
    elasticNetParam=0.0  # L2 only
)

pipeline = Pipeline(stages=[assembler, lr])

# STEP 4: Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"✅ Data split. Train: {train_df.count()} rows, Test: {test_df.count()} rows")

# STEP 5: Fit model
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

# STEP 6: Evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="loan_status_flag",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"✅ Logistic Regression AUC: {auc:.4f}")

spark.stop()
print("✅ Spark session stopped.")
