from typing import Tuple
import pandas as pd
from pydantic import BaseModel
import yaml, json

class DataConfig(BaseModel):
    source: str
    id_column: str
    target: str
    datetime: str

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.source)
    # Basic ordering by snapshot date, if available
    if cfg.datetime in df.columns:
        df[cfg.datetime] = pd.to_datetime(df[cfg.datetime], errors="coerce")
        df = df.sort_values(cfg.datetime)
    return df

def train_valid_split(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds_col = cfg["data"]["datetime"]
    train = df[(df[ds_col] >= cfg["data"]["train_start"]) & (df[ds_col] <= cfg["data"]["train_end"])].copy()
    valid = df[(df[ds_col] >= cfg["data"]["valid_start"]) & (df[ds_col] <= cfg["data"]["valid_end"])].copy()
    return train, valid
