
# spark_logreg_model_final_consolidated.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from scipy.stats import ks_2samp
import pandas as pd

# Start Spark session
spark = SparkSession.builder.appName("LogRegModelFinal").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load data
df = spark.read.csv("../credit_data_aws_flagship2.csv", header=True, inferSchema=True)

# Drop unsupported string columns
df = df.drop("state", "plan_type", "date_issued", "loan_status")

# Encode target variable
loan_indexer = StringIndexer(inputCol="loan_status_flag", outputCol="loan_status_flag")
df = loan_indexer.fit(df).transform(df)

# Handle class imbalance
df = df.withColumn("classWeightCol", when(col("loan_status_flag") == 1, 4.0).otherwise(1.0))

# Define features
feature_cols = [col for col in df.columns if col not in ("loan_status_flag", "classWeightCol", "customer_id")]

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Prepare train/test sets
train_prepared = assembler.transform(train_data).select("features", "loan_status_flag")
test_prepared = assembler.transform(test_data).select("features", "loan_status_flag")

# Train baseline model
print("✅ Training baseline model (no regularization or weighting)...")
lr = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", probabilityCol="probability")
model = lr.fit(train_prepared)
pred_base = model.transform(test_prepared)

# Evaluate AUC
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="probability", metricName="areaUnderROC")
auc_base = evaluator.evaluate(pred_base)
print(f"📊 Baseline Model AUC: {auc_base:.4f}")

# Evaluate KS stat (baseline)
get_prob = udf(lambda v: float(v[1]), DoubleType())
df_ks = pred_base.withColumn("prob_1", get_prob(col("probability"))).select("loan_status_flag", "prob_1").dropna()
ks_base = df_ks.limit(10000).toPandas()
ks_stat, p_val = ks_2samp(ks_base[ks_base["loan_status_flag"] == 1]["prob_1"], ks_base[ks_base["loan_status_flag"] == 0]["prob_1"])
print(f"📉 Baseline KS stat: {ks_stat:.4f} | p-value: {p_val:.4f}")

# Train improved model with weighting and regularization
print("✅ Training improved model (with class weights and regularization)...")
train_data_w, test_data_w = df.randomSplit([0.8, 0.2], seed=42)
train_prep_w = assembler.transform(train_data_w).select("features", "loan_status_flag", "classWeightCol")
test_prep_w = assembler.transform(test_data_w).select("features", "loan_status_flag")

lr_imp = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", weightCol="classWeightCol", regParam=0.1, probabilityCol="probability")
model_imp = lr_imp.fit(train_prep_w)
pred_imp = model_imp.transform(test_prep_w)

auc_imp = evaluator.evaluate(pred_imp)
print(f"📊 Improved Model AUC: {auc_imp:.4f}")

# Evaluate KS stat (improved)
df_ks_imp = pred_imp.withColumn("prob_1", get_prob(col("probability"))).select("loan_status_flag", "prob_1").dropna()
ks_imp = df_ks_imp.limit(10000).toPandas()
ks_stat_imp, p_val_imp = ks_2samp(ks_imp[ks_imp["loan_status_flag"] == 1]["prob_1"], ks_imp[ks_imp["loan_status_flag"] == 0]["prob_1"])
print(f"📉 Improved KS stat: {ks_stat_imp:.4f} | p-value: {p_val_imp:.4f}")

# Display sample predictions
print("📌 Sample predictions from improved model:")
pred_imp.select("fico_score", "loan_amount", "probability", "prediction").show(5)

spark.stop()
