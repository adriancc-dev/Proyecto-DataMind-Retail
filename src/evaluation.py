import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
   
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"
) -> Dict[str, float]:

    metrics = {
        "modelo":    model_name,
        "MAPE (%)":  round(mape(y_true, y_pred), 2),
        "MAE":       round(mae(y_true, y_pred), 2),
        "RMSE":      round(rmse(y_true, y_pred), 2),
    }
    logger.info(
        f"[{model_name:20s}] MAPE={metrics['MAPE (%)']:6.2f}% | "
        f"MAE={metrics['MAE']:8.2f} | RMSE={metrics['RMSE']:8.2f}"
    )
    return metrics


def compare_models(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("modelo").sort_values("MAPE (%)")
    df.insert(0, "Ranking", range(1, len(df) + 1))
    return df
