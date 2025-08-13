# 🚀 Script has started running...
print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.stats import ks_2samp
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("CreditRiskLogReg").getOrCreate()

print("🔄 Loading data...")
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

# Confirm train-test distribution
print("📊 Train-Test Split (80/20)...")
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print("📘 Train Distribution:")
train_data.groupBy("loan_status_flag").count().show()

print("📕 Test Distribution:")
test_data.groupBy("loan_status_flag").count().show()

# Encode categorical column
indexer = StringIndexer(inputCol="util_bin", outputCol="util_bin_index", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["util_bin_index"], outputCols=["util_bin_encoded"])
df_indexed = indexer.fit(df).transform(df)
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# Assemble features
feature_cols = [
    "fico_score", "loan_amount", "monthly_income", "tenure_months",
    "credit_utilization", "has_bankruptcy", "cltv", "high_risk_flag", "util_bin_encoded"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_prepped = assembler.transform(df_encoded).select("features", "loan_status_flag")
train_data, test_data = train_prepped.randomSplit([0.8, 0.2], seed=42)

# Train baseline model
print("💡 Training baseline model (no regularization or weighting)...")
lr = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", probabilityCol="probability")
model = lr.fit(train_data)
pred_base = model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="probability", metricName="areaUnderROC")
auc_base = evaluator.evaluate(pred_base)
print(f"✅ Baseline Model AUC: {auc_base:.4f}")

# KS stat
get_prob = udf(lambda v: float(v[1]), DoubleType())
df_ks = pred_base.withColumn("prob_1", get_prob(col("probability")))                  .select("loan_status_flag", "prob_1").dropna()
sample_ks = df_ks.limit(1000).toPandas()
ks_stat, p_val = ks_2samp(sample_ks["prob_1"], sample_ks["loan_status_flag"])
print(f"📉 Baseline KS: {ks_stat:.4f} | p-value: {p_val:.4f}")

# Train improved model
print("🚀 Training improved model (classWeightCol + regParam=0.1)...")
df_encoded = df_encoded.withColumn("classWeightCol", when(col("loan_status_flag") == 1, 4.0).otherwise(1.0))
train_prepped_weighted = assembler.transform(df_encoded).select("features", "loan_status_flag", "classWeightCol")
train_data_w, test_data_w = train_prepped_weighted.randomSplit([0.8, 0.2], seed=42)

lr_imp = LogisticRegression(featuresCol="features", labelCol="loan_status_flag",
                            weightCol="classWeightCol", regParam=0.1, probabilityCol="probability")
model_imp = lr_imp.fit(train_data_w)
pred_imp = model_imp.transform(test_data_w)

auc_imp = evaluator.evaluate(pred_imp)
print(f"✅ Improved Model AUC: {auc_imp:.4f}")

# KS stat (improved)
df_ks_imp = pred_imp.withColumn("prob_1", get_prob(col("probability")))                     .select("loan_status_flag", "prob_1").dropna()
sample_ks_imp = df_ks_imp.limit(1000).toPandas()
ks_stat_imp, p_val_imp = ks_2samp(sample_ks_imp["prob_1"], sample_ks_imp["loan_status_flag"])
print(f"📉 Improved KS: {ks_stat_imp:.4f} | p-value: {p_val_imp:.4f}")

# Sample predictions
print("📌 Sample predictions (Improved Model):")
pred_imp.select("fico_score", "loan_amount", "probability", "prediction").show(5)

spark.stop()
