
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 1. Initialize Spark
spark = SparkSession.builder.appName("Credit Risk Modeling").getOrCreate()

# 2. Load cleaned data
df = spark.read.parquet("../data/processed_spark/credit_data_clean.parquet")
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

# 7. Evaluate performance
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"✅ Model AUC: {auc:.4f}")

# 8. Optional: Show predictions
predictions.select("fico_score", "loan_amount", "probability", "prediction").show(5)

# 9. Stop Spark session
spark.stop()
