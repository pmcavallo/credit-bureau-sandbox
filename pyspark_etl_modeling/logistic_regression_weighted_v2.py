# 🚀 Script has started running...
print("🚀 Script has started running...")
"""
Logistic Regression with Class Weights, Regularization, and KS Evaluation
For Credit Risk Modeling using PySpark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Start Spark session with increased memory
spark = SparkSession.builder     .appName("CreditRiskLogisticRegression")     .config("spark.driver.memory", "4g")     .getOrCreate()

print("✅ Spark session started.")

# Load the transformed data (adjust path if needed)
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")
#print(f"✅ Loaded transformed data. Row count: {df.count()}")
print("✅ Data split done.")

# STEP 1: Create class weights
label_counts = df.groupBy("loan_status_flag").count().toPandas()
majority = label_counts["count"].max()
weights = {
    int(row["loan_status_flag"]): majority / row["count"]
    for _, row in label_counts.iterrows()
}

from pyspark.sql.functions import when

df = df.withColumn(
    "classWeightCol",
    when(col("loan_status_flag") == 1, float(weights[1]))
    .otherwise(float(weights[0]))
)

print(f"✅ Added classWeightCol. Weights: {weights}")

# STEP 2: Assemble features
features = ['fico_score', 'loan_amount', 'cltv', 'delinq_flag', 'high_risk_flag']
assembler = VectorAssembler(inputCols=features, outputCol="features")

# STEP 3: Logistic regression with regularization
lr = LogisticRegression(
    labelCol="loan_status_flag",
    featuresCol="features",
    weightCol="classWeightCol",
    regParam=0.1,
    elasticNetParam=0.0  # L2 only
)

pipeline = Pipeline(stages=[assembler, lr])

# STEP 4: Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print("✅ Data split done.")  # Count logging removed to prevent Java gateway crash

# STEP 5: Fit model
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

# STEP 6: Evaluate AUC
evaluator = BinaryClassificationEvaluator(
    labelCol="loan_status_flag",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"✅ Logistic Regression AUC: {auc:.4f}")

# STEP 7: KS calculation
# Get prediction probabilities and true labels
pred_df = predictions.select("probability", "loan_status_flag")     .withColumn("prob_default", col("probability").getItem(1))

# Create deciles
from pyspark.sql.window import Window
from pyspark.sql.functions import ntile, sum as spark_sum

windowSpec = Window.orderBy(col("prob_default").desc())
ks_df = pred_df.withColumn("decile", ntile(10).over(windowSpec))

# Aggregate by decile
agg_df = ks_df.groupBy("decile").agg(
    spark_sum((col("loan_status_flag") == 1).cast("int")).alias("bads"),
    spark_sum((col("loan_status_flag") == 0).cast("int")).alias("goods")
).orderBy("decile")

# Calculate cumulative bads/goods and KS
from pyspark.sql.functions import lit

total = agg_df.selectExpr("sum(bads) as total_bads", "sum(goods) as total_goods").collect()[0]
total_bads = total["total_bads"]
total_goods = total["total_goods"]

agg_df = agg_df.withColumn("cum_bads", spark_sum("bads").over(Window.orderBy("decile")))
agg_df = agg_df.withColumn("cum_goods", spark_sum("goods").over(Window.orderBy("decile")))
agg_df = agg_df.withColumn("cum_bad_pct", col("cum_bads") / lit(total_bads))
agg_df = agg_df.withColumn("cum_good_pct", col("cum_goods") / lit(total_goods))
agg_df = agg_df.withColumn("ks", (col("cum_bad_pct") - col("cum_good_pct")).cast("double"))

ks_value = agg_df.agg({"ks": "max"}).collect()[0][0]
print(f"✅ KS Statistic: {ks_value:.4f}")

spark.stop()
print("✅ Spark session stopped.")
