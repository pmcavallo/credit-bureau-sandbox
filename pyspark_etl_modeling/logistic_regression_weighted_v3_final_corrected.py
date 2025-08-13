import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from scipy.stats import ks_2samp

# Start Spark session
spark = SparkSession.builder.appName("CreditRiskLogisticRegressionV3FinalCorrected").getOrCreate()
print("🚀 Script has started running...")
print("✅ Spark session started.")

# Load data
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

# Filter out string columns (corrected)
features = ['fico_score', 'loan_amount_log', 'monthly_income_log', 'cltv_log', 'tenure_months', 'credit_utilization', 'has_bankruptcy', 'plan_type_ohe', 'util_bin_ohe', 'cltv_log_bin']

# Assemble features
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print("✅ Data split done.")

# Logistic regression with class weights
pos = train_data.filter("label == 1").count()
neg = train_data.filter("label == 0").count()
balancing_ratio = neg / pos
train_data = train_data.withColumn("classWeightCol", when(col("label") == 1, balancing_ratio).otherwise(1.0))
lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="classWeightCol")
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Evaluate AUC
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"📈 Logistic Regression AUC: {auc:.4f}")

# KS Statistic + p-value
pred_df = predictions.select("probability", "label").withColumn("prob_default", col("probability").getItem(1)).select("prob_default", "label")
pandas_df = pred_df.toPandas()
ks_stat, p_value = ks_2samp(pandas_df[pandas_df['label'] == 1]['prob_default'],
                            pandas_df[pandas_df['label'] == 0]['prob_default'])
print(f"📊 KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4g}")

# Stop Spark
spark.stop()
print("✅ Spark session stopped.")
