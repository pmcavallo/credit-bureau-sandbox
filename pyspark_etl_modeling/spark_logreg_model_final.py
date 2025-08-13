
# spark_logreg_model_final.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType
from scipy.stats import ks_2samp

# Initialize Spark
spark = SparkSession.builder.appName("CreditRiskLogRegFinal").getOrCreate()

# Load cleaned Parquet data
df = spark.read.parquet("output/credit_data_cleaned2.parquet")

# Split train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Assemble features
feature_cols = [
    "fico_score", "loan_amount", "tenure_months", "monthly_income",
    "cltv", "has_bankruptcy", "delinq_flag", "high_risk_flag",
    "util_bin_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_assembled = assembler.transform(train_df)
test_assembled = assembler.transform(test_df)

# Baseline Model
lr_base = LogisticRegression(featuresCol="features", labelCol="loan_status_flag")
model_base = lr_base.fit(train_assembled)
pred_base = model_base.transform(test_assembled)

# Evaluate AUC
evaluator = BinaryClassificationEvaluator(labelCol="loan_status_flag")
auc_base = evaluator.evaluate(pred_base)

# Evaluate KS
def get_ks(df, label_col="loan_status_flag", prob_col="probability"):
    get_prob = lambda v: float(v[1])
    prob_udf = spark.udf.register("get_prob", get_prob, DoubleType())
    df_prob = df.withColumn("prob_1", prob_udf(col(prob_col)))                 .select("prob_1", label_col)                 .dropna()
    pdf = df_prob.limit(1000).toPandas()
    ks_stat, p_val = ks_2samp(pdf["prob_1"], pdf[label_col])
    return ks_stat, p_val

ks_base, p_base = get_ks(pred_base)

# Improved Model (Weight + Regularization)
train_weighted = train_assembled.withColumn(
    "classWeightCol",
    when(col("loan_status_flag") == 1, 4.0).otherwise(1.0)
)

lr_weighted = LogisticRegression(
    featuresCol="features", labelCol="loan_status_flag",
    weightCol="classWeightCol", regParam=0.1
)

model_weighted = lr_weighted.fit(train_weighted)
pred_weighted = model_weighted.transform(test_assembled)

auc_weighted = evaluator.evaluate(pred_weighted)
ks_weighted, p_weighted = get_ks(pred_weighted)

# Print Metrics
print(f"✅ Baseline Model AUC: {auc_base:.4f}")
print(f"✅ Baseline KS: {ks_base:.4f} | p-value: {p_base:.4f}")
print(f"✅ Improved Model AUC: {auc_weighted:.4f}")
print(f"✅ Improved KS: {ks_weighted:.4f} | p-value: {p_weighted:.4f}")

# Sample Predictions
pred_weighted.select("fico_score", "loan_amount", "probability", "prediction").show(5)
