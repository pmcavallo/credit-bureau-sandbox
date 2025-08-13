import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType

# Initialize Spark
spark = SparkSession.builder.appName("CreditRiskLogReg").getOrCreate()

# Load cleaned data
df = spark.read.parquet("output/credit_data_cleaned2.parquet")

# Encode categorical feature 'util_bin'
indexer = StringIndexer(inputCol="util_bin", outputCol="util_bin_index")
df = indexer.fit(df).transform(df)

# Assemble features
feature_cols = [
    "fico_score", "loan_amount", "tenure_months", "monthly_income",
    "credit_utilization", "has_bankruptcy", "cltv", "delinq_flag",
    "high_risk_flag", "util_bin_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Train baseline logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", predictionCol="prediction", probabilityCol="probability")
baseline_model = lr.fit(train_df)

# Evaluate
predictions = baseline_model.transform(test_df)

evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

print(f"✅ Baseline Model AUC: {auc:.4f}")
