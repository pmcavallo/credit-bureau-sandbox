from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_transformer(cat: List[str], num: List[str], passthrough: List[str] = None) -> ColumnTransformer:
    if passthrough is None: passthrough = []
    transformers = []
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    if num:
        transformers.append(("num", StandardScaler(), num))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def fit_transform(ct: ColumnTransformer, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    X = ct.fit_transform(df)
    # For simplicity, return as DataFrame when feasible
    try:
        feature_names = []
        for name, trans, cols in ct.transformers_:
            if name == "cat":
                feature_names += list(ct.named_transformers_["cat"].get_feature_names_out(cols))
            elif name == "num":
                feature_names += cols
        X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=feature_names, index=df.index)
        return X_df, feature_names
    except Exception:
        return pd.DataFrame(X), []
