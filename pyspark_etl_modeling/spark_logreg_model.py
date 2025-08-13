print("🚀 Script has started running...")

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from scipy.stats import ks_2samp
from pyspark.ml.functions import vector_to_array  # ✅ add this




# 1. Initialize Spark
spark = SparkSession.builder.appName("Credit Risk Modeling").getOrCreate()

# 2. Load cleaned data
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

print("✅ Cleaned data loaded")

# 3. Assemble features
feature_cols = ['fico_score', 'loan_amount']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df)

# 4. Train/test split
train_df, test_df = df_vector.randomSplit([0.7, 0.3], seed=42)

# 5. Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="loan_status_flag")
lr_model = lr.fit(train_df)

# 6. Predict on test set
predictions = lr_model.transform(test_df)

# 7. Evaluate performance (AUC)
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"✅ Model AUC: {auc:.4f}")

# 8. Optional: Show predictions
predictions.select("fico_score", "loan_amount", "probability", "prediction").show(5)

# 9. KS Statistic Calculation
df_ks = predictions.withColumn("prob_1", vector_to_array(col("probability"))[1]) \
    .select("loan_status_flag", "prob_1") \
    .dropna()

# Convert to Pandas
pandas_df = df_ks.toPandas()
good = pandas_df[pandas_df['loan_status_flag'] == 0]['prob_1']
bad = pandas_df[pandas_df['loan_status_flag'] == 1]['prob_1']

ks_stat, p_value = ks_2samp(good, bad)
print(f"✅ KS Statistic: {ks_stat:.4f} | p-value: {p_value:.4f}")

# 10. Stop Spark session
spark.stop()
