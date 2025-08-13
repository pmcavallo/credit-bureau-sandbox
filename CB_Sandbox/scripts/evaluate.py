import argparse, yaml, os, json
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.bureau_pipeline.data_prep import DataConfig, load_config, load_data, train_valid_split
from src.bureau_pipeline.features import build_transformer, fit_transform
from src.bureau_pipeline.modeling import train_models
from src.bureau_pipeline.eval import ks_score
from src.bureau_pipeline.monitoring import psi
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
X_tr, _ = fit_transform(ct, train_df)
X_va, _ = fit_transform(ct, valid_df)

models = train_models(X_tr, y_tr, class_weight=cfg["modeling"].get("class_weight", None))
best = max(models, key=lambda k: roc_auc_score(y_va, models[k].predict_proba(X_va)[:,1]))
p_tr = models[best].predict_proba(X_tr)[:,1]
p_va = models[best].predict_proba(X_va)[:,1]

metrics = {
    "best_model": best,
    "auc_train": float(roc_auc_score(y_tr, p_tr)),
    "auc_valid": float(roc_auc_score(y_va, p_va)),
    "ks_valid": ks_score(y_va, p_va),
    "psi_valid_vs_train": psi(pd.Series(p_tr), pd.Series(p_va))
}

os.makedirs(cfg["paths"]["metrics"], exist_ok=True)
save_json(metrics, os.path.join(cfg["paths"]["metrics"], "evaluation.json"))
print("Evaluation complete. Saved to outputs/metrics/evaluation.json")
