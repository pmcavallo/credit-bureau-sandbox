# 🚀 Script has started running...
print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from scipy.stats import ks_2samp

# Initialize Spark session
spark = SparkSession.builder.appName("LogisticRegressionCreditRisk").getOrCreate()

# Load data
data = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

# Create util_bin_index (from existing util_bin)
indexer = StringIndexer(inputCol="util_bin", outputCol="util_bin_index")
data = indexer.fit(data).transform(data)

# Assemble features
feature_cols = [
    "fico_score", "loan_amount", "tenure_months", "state", "plan_type",
    "monthly_income", "date_issued", "loan_status", "loan_status_flag",
    "credit_utilization", "has_bankruptcy", "cltv", "util_bin_index",
    "delinq_flag", "high_risk_flag"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
train_prepared = assembler.transform(train_data).select("features", "loan_status_flag")
test_base = assembler.transform(test_data).select("features", "loan_status_flag")

# Train baseline model
print("💡 Training baseline model (no regularization or weighting)...")
lr = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", probabilityCol="probability")
model = lr.fit(train_prepared)
pred_base = model.transform(test_base)

# Evaluate baseline model
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="probability", metricName="areaUnderROC")
auc_base = evaluator.evaluate(pred_base)
print(f"✅ Baseline Model AUC: {auc_base:.4f}")

# KS statistic function
get_prob = udf(lambda v: float(v[1]), DoubleType())
df_ks = pred_base.withColumn("prob_1", get_prob(col("probability")))     .select("loan_status_flag", "prob_1").dropna()
sample_ks = df_ks.limit(1000).toPandas()
ks_stat, p_val = ks_2samp(sample_ks["prob_1"], sample_ks["loan_status_flag"])
print(f"📊 Baseline KS: {ks_stat:.4f} | p-value: {p_val:.4f}")

# Train improved model (with weights + regularization)
print("🔁 Training improved model (classWeightCol + regParam=0.1)...")
df_weighted = data.withColumn("classWeightCol", (col("loan_status_flag") == 1).cast("double") * 4.0 + 1.0)
train_data_w, test_data_w = df_weighted.randomSplit([0.8, 0.2], seed=42)
train_weighted = assembler.transform(train_data_w).select("features", "loan_status_flag", "classWeightCol")
test_weighted = assembler.transform(test_data_w).select("features", "loan_status_flag")

lr_imp = LogisticRegression(featuresCol="features", labelCol="loan_status_flag",
                            weightCol="classWeightCol", regParam=0.1, probabilityCol="probability")
model_imp = lr_imp.fit(train_weighted)
pred_imp = model_imp.transform(test_weighted)

auc_imp = evaluator.evaluate(pred_imp)
print(f"✅ Improved Model AUC: {auc_imp:.4f}")

# KS stat (improved)
df_ks_imp = pred_imp.withColumn("prob_1", get_prob(col("probability")))     .select("loan_status_flag", "prob_1").dropna()
sample_ks_imp = df_ks_imp.limit(1000).toPandas()
ks_stat_imp, p_val_imp = ks_2samp(sample_ks_imp["prob_1"], sample_ks_imp["loan_status_flag"])
print(f"📊 Improved KS: {ks_stat_imp:.4f} | p-value: {p_val_imp:.4f}")

# View some predictions
print("📌 Sample predictions (Improved Model):")
pred_imp.select("fico_score", "loan_amount", "probability", "prediction").show(5)
