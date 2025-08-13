from typing import Dict, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_models(X: pd.DataFrame, y: pd.Series, class_weight='balanced') -> Dict[str, object]:
    models = {}
    models["logistic_regression"] = LogisticRegression(max_iter=2000, class_weight=class_weight, n_jobs=None)
    models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0, eval_metric="logloss")
    models["lightgbm"] = LGBMClassifier(n_estimators=500, max_depth=-1, subsample=0.9, colsample_bytree=0.9, reg_lambda=5.0)

    for name, m in models.items():
        m.fit(X, y)
    return models

def score_auc(models: Dict[str, object], X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    out = {}
    for name, m in models.items():
        p = m.predict_proba(X)[:,1]
        out[name] = float(roc_auc_score(y, p))
    return out
