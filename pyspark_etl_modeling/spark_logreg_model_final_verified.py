
# 🚀 Script has started running...
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, struct
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
import numpy as np

spark = SparkSession.builder.appName("CreditRiskLogReg").getOrCreate()

# ✅ Load cleaned Parquet data
df = spark.read.parquet("C:/credit_risk_project/pyspark_etl_modeling/output/credit_data_cleaned2.parquet")

# ✅ Recast columns as needed
df = df.withColumn("fico_score", col("fico_score").cast(DoubleType())) \
       .withColumn("loan_amount", col("loan_amount").cast(DoubleType())) \
       .withColumn("tenure_months", col("tenure_months").cast(DoubleType())) \
       .withColumn("monthly_income", col("monthly_income").cast(DoubleType())) \
       .withColumn("credit_utilization", col("credit_utilization").cast(DoubleType())) \
       .withColumn("has_bankruptcy", col("has_bankruptcy").cast(DoubleType())) \
       .withColumn("cltv", col("cltv").cast(DoubleType())) \
       .withColumn("delinq_flag", col("delinq_flag").cast(DoubleType())) \
       .withColumn("high_risk_flag", col("high_risk_flag").cast(DoubleType())) \
       .withColumn("loan_status_flag", col("loan_status_flag").cast(IntegerType()))

# ✅ Feature assembly
features = ["fico_score", "loan_amount", "tenure_months", "monthly_income", 
            "credit_utilization", "has_bankruptcy", "cltv", "delinq_flag", "high_risk_flag"]

assembler = VectorAssembler(inputCols=features, outputCol="features")
df_prepared = assembler.transform(df)

# ✅ Train-test split
train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)

# ✅ Train baseline model (no regularization or class weight)
print("\n🟡 Training baseline model...")
lr_base = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", predictionCol="prediction", probabilityCol="probability")
model_base = lr_base.fit(train)
preds_base = model_base.transform(test)

# ✅ Evaluate baseline
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag", rawPredictionCol="rawPrediction")
auc_base = evaluator.evaluate(preds_base)
ks_base = ks_2samp(
    preds_base.filter("loan_status_flag = 1").select("probability").rdd.map(lambda x: x[0][1]).collect(),
    preds_base.filter("loan_status_flag = 0").select("probability").rdd.map(lambda x: x[0][1]).collect()
)

print(f"✅ Baseline Model AUC: {auc_base:.4f}")
print(f"✅ Baseline KS: {ks_base.statistic:.4f} | p-value: {ks_base.pvalue:.4f}")

# ✅ Train improved model with class weight and regularization
print("\n🟡 Training improved model (classWeightCol + regParam=0.1)...")
train = train.withColumn("classWeightCol", when(col("loan_status_flag") == 1, 4.0).otherwise(1.0))

lr_imp = LogisticRegression(featuresCol="features", labelCol="loan_status_flag", weightCol="classWeightCol", regParam=0.1)
model_imp = lr_imp.fit(train)
preds_imp = model_imp.transform(test)

# ✅ Evaluate improved model
auc_imp = evaluator.evaluate(preds_imp)
ks_imp = ks_2samp(
    preds_imp.filter("loan_status_flag = 1").select("probability").rdd.map(lambda x: x[0][1]).collect(),
    preds_imp.filter("loan_status_flag = 0").select("probability").rdd.map(lambda x: x[0][1]).collect()
)

print(f"✅ Improved Model AUC: {auc_imp:.4f}")
print(f"✅ Improved KS: {ks_imp.statistic:.4f} | p-value: {ks_imp.pvalue:.4f}")

# ✅ Show sample predictions
print("\n📦 Sample predictions from improved model:")
preds_imp.select("fico_score", "loan_amount", "probability", "prediction").show(5)
