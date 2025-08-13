# 🚀 Script has started running...
print("🚀 Script has started running...")
# spark_logreg_model_merged.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, lit, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from scipy.stats import ks_2samp
import pandas as pd

# --------------------------
# 💻 Spark Session
# --------------------------
spark = SparkSession.builder.appName("LogRegCreditRisk").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --------------------------
# 📥 Load Parquet
# --------------------------
print("📦 Loading data...")
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")
df = df.dropna()

label_col = "loan_status_flag"
features = ["fico_score", "loan_amount"]

# --------------------------
# 🧼 Feature Engineering
# --------------------------
va = VectorAssembler(inputCols=features, outputCol="features")
df = va.transform(df)

# --------------------------
# 📊 Data Split
# --------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)
print("📘 Train Distribution:")
train.groupBy(label_col).agg(count("*").alias("count")).show()

print("📕 Test Distribution:")
test.groupBy(label_col).agg(count("*").alias("count")).show()


# --------------------------
# 🧪 Train Baseline Model
# --------------------------
print("🔽 Training baseline model (no regularization or weighting)...")
baseline_lr = LogisticRegression(labelCol=label_col, featuresCol="features", predictionCol="prediction", probabilityCol="probability")
baseline_model = baseline_lr.fit(train)
baseline_predictions = baseline_model.transform(test)

evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
baseline_auc = evaluator.evaluate(baseline_predictions)
print(f"✅ Baseline Model AUC: {baseline_auc:.4f}")

# Extract probability of class 1
get_prob = udf(lambda v: float(v[1]), DoubleType())
baseline_predictions = baseline_predictions.withColumn("prob_1", get_prob(col("probability")))

# KS statistic for baseline
df_ks = baseline_predictions.select(label_col, "prob_1").dropna()
sample_ks = df_ks.limit(1000).toPandas()
ks_stat, p_val = ks_2samp(sample_ks["prob_1"], sample_ks[label_col])
print(f"📊 Baseline KS: {ks_stat:.4f} | p-value: {p_val:.4f}")

# --------------------------
# 📈 Train Improved Model
# --------------------------
print("🔼 Training improved model (classWeightCol + regParam=0.1)...")

# Add weight column: assign higher weight to minority class
train = train.withColumn("weight", when(col(label_col) == 1, 3.0).otherwise(1.0))

train.groupBy(label_col).agg(count("*").alias("count"), 
                             {"weight": "avg"}).show()

improved_lr = LogisticRegression(
    labelCol=label_col,
    featuresCol="features",
    predictionCol="prediction",
    probabilityCol="probability",
    weightCol="weight",
    regParam=0.1
)
improved_model = improved_lr.fit(train)
improved_predictions = improved_model.transform(test)

improved_auc = evaluator.evaluate(improved_predictions)
print(f"✅ Improved Model AUC: {improved_auc:.4f}")

# KS statistic for improved model
improved_predictions = improved_predictions.withColumn("prob_1", get_prob(col("probability")))
df_ks_imp = improved_predictions.select(label_col, "prob_1").dropna()
sample_ks_imp = df_ks_imp.limit(1000).toPandas()
ks_stat_imp, p_val_imp = ks_2samp(sample_ks_imp["prob_1"], sample_ks_imp[label_col])
print(f"📊 Improved KS: {ks_stat_imp:.4f} | p-value: {p_val_imp:.4f}")

# --------------------------
# 🧾 Show Output
# --------------------------
print("📌 Sample predictions (Improved Model):")
improved_predictions.select("fico_score", "loan_amount", "probability", "prediction").show(5, truncate=False)
