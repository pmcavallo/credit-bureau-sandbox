
# logistic_regression_weighted_v3_final.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
import warnings
warnings.filterwarnings("ignore")

# Start Spark session
spark = SparkSession.builder.appName("CreditRiskLogisticRegressionFinal") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("🪄 Script has started running...")
print("✅ Spark session started.")

# Load the cleaned data
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

# Define label and features
label_col = "target"
feature_cols = [c for c in df.columns if c not in [label_col]]

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Compute class weights
class_counts = df.groupBy(label_col).count().collect()
counts_dict = {row[label_col]: row["count"] for row in class_counts}
majority = max(counts_dict.values())
class_weights = {k: float(majority) / v for k, v in counts_dict.items()}
print(f"✅ Added classWeightCol. Weights: {class_weights}")

# Add weight column
df = df.withColumn("classWeightCol", when(col(label_col) == 1, class_weights[1]).otherwise(class_weights[0]))

# Train/test split
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
print("✅ Data split done.")

# Fit model with weights
lr = LogisticRegression(featuresCol="features", labelCol=label_col, weightCol="classWeightCol", probabilityCol="probability")
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Evaluate AUC
evaluator = BinaryClassificationEvaluator(labelCol=label_col)
auc = evaluator.evaluate(predictions)
print(f"📈 Logistic Regression AUC: {auc:.4f}")

# Extract predicted probabilities
pred_df = predictions.select("probability", label_col) \
    .withColumn("prob_default", col("probability").getItem(1)) \
    .select("prob_default", label_col)

# Calculate KS using Spark
window = F.window.orderBy(col("prob_default").desc())
pos = pred_df.filter(col(label_col) == 1).count()
neg = pred_df.filter(col(label_col) == 0).count()

ks_df = pred_df.withColumn("TPR", (col(label_col).cast("double").cumsum() / pos)) \
               .withColumn("FPR", ((1 - col(label_col).cast("double")).cumsum() / neg)) \
               .withColumn("KS", col("TPR") - col("FPR"))

ks_stat = ks_df.agg(F.max("KS")).collect()[0][0]

# KS p-value (pandas + scipy)
from scipy.stats import ks_2samp
pdf_sample = pred_df.sample(fraction=0.2, seed=42).toPandas()
score_1 = pdf_sample.loc[pdf_sample[label_col] == 1, "prob_default"]
score_0 = pdf_sample.loc[pdf_sample[label_col] == 0, "prob_default"]
ks_stat_scipy, ks_pvalue = ks_2samp(score_1, score_0)

print(f"✅ KS Statistic: {ks_stat_scipy:.4f}")
print(f"📌 KS p-value: {ks_pvalue:.4e}")

spark.stop()
print("✅ Spark session stopped.")
