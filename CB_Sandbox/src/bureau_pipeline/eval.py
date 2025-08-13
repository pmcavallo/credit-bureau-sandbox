import numpy as np
import pandas as pd

def ks_score(y_true: pd.Series, y_proba: np.ndarray, bins: int = 20) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grouped = df.groupby("bucket").agg(event_rate=("y", "mean"), count=("y", "size")).sort_index()
    cum_event = (grouped["event_rate"] * grouped["count"]).cumsum() / (df["y"].sum() + 1e-9)
    cum_non_event = ((1 - grouped["event_rate"]) * grouped["count"]).cumsum() / ((len(df) - df["y"].sum()) + 1e-9)
    ks = (cum_event - cum_non_event).abs().max()
    return float(ks)
