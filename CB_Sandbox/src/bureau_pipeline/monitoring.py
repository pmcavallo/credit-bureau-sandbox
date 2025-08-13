import pandas as pd
import numpy as np

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e_perc, a_perc = [], []
    cuts = np.quantile(expected.dropna(), q=np.linspace(0,1,bins+1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=cuts)[0]
    a_counts = np.histogram(actual, bins=cuts)[0]
    e_perc = e_counts / (e_counts.sum() + 1e-9)
    a_perc = a_counts / (a_counts.sum() + 1e-9)
    v = (a_perc - e_perc) * np.log((a_perc + 1e-9) / (e_perc + 1e-9))
    return float(np.sum(v))
