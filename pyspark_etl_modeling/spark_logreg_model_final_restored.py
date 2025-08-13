# 🚀 Script has started running...
print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

# Start Spark session
spark = SparkSession.builder.appName("CreditRiskLogReg").getOrCreate()
print("🚀 Script has started running...")

# Load cleaned Parquet data
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")
print("📦 Data loaded from cleaned Parquet file")

# String index label column
label_indexer = StringIndexer(inputCol="loan_status_flag", outputCol="loan_status_flag_index")
df = label_indexer.fit(df).transform(df)

# Feature columns
feature_cols = [col for col in df.columns if col not in ["customer_id", "loan_status_flag", "loan_status_flag_index"]]

# Vector assembler
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Split train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print("📊 Train-Test Split (80/20)...")
print("📈 Train Distribution:")
train_df.groupBy("loan_status_flag").count().show()
print("🧪 Test Distribution:")
test_df.groupBy("loan_status_flag").count().show()

# Train baseline model
print("⚙️ Training baseline model (no regularization or weighting)...")
baseline = LogisticRegression(featuresCol="features", labelCol="loan_status_flag_index", predictionCol="prediction", probabilityCol="probability")
baseline_model = baseline.fit(train_df)
baseline_preds = baseline_model.transform(test_df)

# Evaluate baseline
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
baseline_auc = evaluator.evaluate(baseline_preds)
baseline_probs = np.array(baseline_preds.select("probability").rdd.map(lambda row: float(row["probability"][1])).collect())
baseline_labels = np.array(baseline_preds.select("loan_status_flag_index").rdd.map(float).collect())
ks_stat_base, p_val_base = ks_2samp(baseline_probs[baseline_labels == 1], baseline_probs[baseline_labels == 0])
print(f"✅ Baseline Model AUC: {baseline_auc:.4f}")
print(f"✅ Baseline KS: {ks_stat_base:.4f} | p-value: {p_val_base:.4f}")

# Train improved model with class weights and regularization
print("⚙️ Training improved model (classWeightCol + regParam=0.1)...")
class_weights = train_df.groupBy("loan_status_flag_index").count().toPandas()
total = class_weights["count"].sum()
weights_dict = {row["loan_status_flag_index"]: total / row["count"] for _, row in class_weights.iterrows()}
weighted_df = train_df.withColumn("classWeightCol", col("loan_status_flag_index").cast("int").cast("double").apply(lambda x: float(weights_dict[x])))

improved = LogisticRegression(featuresCol="features", labelCol="loan_status_flag_index", weightCol="classWeightCol", regParam=0.1, predictionCol="prediction", probabilityCol="probability")
improved_model = improved.fit(weighted_df)
improved_preds = improved_model.transform(test_df)

# Evaluate improved
improved_auc = evaluator.evaluate(improved_preds)
improved_probs = np.array(improved_preds.select("probability").rdd.map(lambda row: float(row["probability"][1])).collect())
improved_labels = np.array(improved_preds.select("loan_status_flag_index").rdd.map(float).collect())
ks_stat_imp, p_val_imp = ks_2samp(improved_probs[improved_labels == 1], improved_probs[improved_labels == 0])
print(f"✅ Improved Model AUC: {improved_auc:.4f}")
print(f"✅ Improved KS: {ks_stat_imp:.4f} | p-value: {p_val_imp:.4f}")

# Show predictions
print("📦 Sample predictions from improved model:")
improved_preds.select("fico_score", "loan_amount", "probability", "prediction").show(5)

spark.stop()
