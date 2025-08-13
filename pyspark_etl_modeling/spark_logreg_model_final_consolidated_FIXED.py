# 🚀 Script has started running...
print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName("LogRegCreditRisk").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load cleaned parquet data
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

# Create util_bin and util_bin_index
df = df.withColumn("util_bin", col("credit_utilization").cast("double"))
df = df.withColumn("util_bin_index", when(col("util_bin") < 0.3, 0)
                                   .when(col("util_bin") < 0.7, 1)
                                   .otherwise(2))

# Create delinq_flag from high_risk_flag if needed
if "delinq_flag" not in df.columns:
    df = df.withColumn("delinq_flag", col("high_risk_flag"))

# Index label column
loan_indexer = StringIndexer(inputCol="loan_status_flag", outputCol="label")
df = loan_indexer.fit(df).transform(df)

# Assemble features
feature_cols = ["fico_score", "loan_amount", "tenure_months", "monthly_income",
                "credit_utilization", "has_bankruptcy", "cltv", "util_bin_index", "delinq_flag", "high_risk_flag"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_prepared = assembler.transform(df)

# Train/test split
train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)

# Train baseline model (no weighting)
lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability")
model = lr.fit(train)
pred_base = model.transform(test)

# Evaluate baseline model
get_prob = udf(lambda v: float(v[1]), DoubleType())
df_ks = pred_base.withColumn("prob_1", get_prob(col("probability"))).select("loan_status_flag", "prob_1").dropna()
ks_base = df_ks.limit(10000).toPandas()
ks_stat, p_val = ks_2samp(ks_base[ks_base["loan_status_flag"]==1]["prob_1"], ks_base[ks_base["loan_status_flag"]==0]["prob_1"])
y_true = ks_base["loan_status_flag"]
y_score = ks_base["prob_1"]
auc = roc_auc_score(y_true, y_score)
print("✅ Baseline Model AUC: {:.4f}".format(auc))
print(f"📊 Baseline KS stat: {ks_stat:.4f} | p-value: {p_val:.4f}")

# Train improved model with weighting and regularization
print("✅ Training improved model (with class weights and regularization)...")
lr_imp = LogisticRegression(featuresCol="features", labelCol="label", weightCol="classWeightCol",
                            maxIter=10, regParam=0.1, elasticNetParam=0.8, predictionCol="prediction", probabilityCol="probability")
# Create class weights
major = df.filter("label == 0").count()
minor = df.filter("label == 1").count()
balancing_ratio = minor / (major + minor)
df = df.withColumn("classWeightCol", when(col("label") == 1, 1 - balancing_ratio).otherwise(balancing_ratio))

df_prepared = assembler.transform(df)
train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)
model_imp = lr_imp.fit(train)
pred_imp = model_imp.transform(test)

# Evaluate improved model
df_ks = pred_imp.withColumn("prob_1", get_prob(col("probability"))).select("loan_status_flag", "prob_1").dropna()
ks_imp = df_ks.limit(10000).toPandas()
ks_stat, p_val = ks_2samp(ks_imp[ks_imp["loan_status_flag"]==1]["prob_1"], ks_imp[ks_imp["loan_status_flag"]==0]["prob_1"])
y_true = ks_imp["loan_status_flag"]
y_score = ks_imp["prob_1"]
auc = roc_auc_score(y_true, y_score)
print("✅ Improved Model AUC: {:.4f}".format(auc))
print(f"📊 Improved KS stat: {ks_stat:.4f} | p-value: {p_val:.4f}")

# Show sample predictions
print("✅ Sample predictions from improved model:")
pred_imp.select("fico_score", "loan_amount", "probability", "prediction").show(5)
