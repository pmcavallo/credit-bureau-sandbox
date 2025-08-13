import argparse, yaml, os, json
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.bureau_pipeline.data_prep import DataConfig, load_config, load_data, train_valid_split
from src.bureau_pipeline.features import build_transformer, fit_transform
from src.bureau_pipeline.modeling import train_models, score_auc
from src.bureau_pipeline.utils import save_json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/base.yaml')
args = parser.parse_args()

cfg = load_config(args.config)
dc = DataConfig(**cfg["data"])
df = load_data(dc)

train_df, valid_df = train_valid_split(df, cfg)
y_tr = train_df[cfg["data"]["target"]]
y_va = valid_df[cfg["data"]["target"]]

ct = build_transformer(cfg["features"]["categorical"], cfg["features"]["numeric"], cfg["features"].get("passthrough", []))

X_tr, feature_names = fit_transform(ct, train_df)
X_va, _ = fit_transform(ct, valid_df)

models = train_models(X_tr, y_tr, class_weight=cfg["modeling"].get("class_weight", None))
auc_tr = score_auc(models, X_tr, y_tr)
auc_va = score_auc(models, X_va, y_va)

os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
save_json({"auc_train": auc_tr, "auc_valid": auc_va, "features": feature_names}, os.path.join(cfg["paths"]["metrics"], "metrics.json"))

print("Training complete. Metrics written to", os.path.join(cfg["paths"]["metrics"], "metrics.json"))
