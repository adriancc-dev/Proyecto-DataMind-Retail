import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

FESTIVOS_ES = pd.to_datetime([
    "2022-01-01", "2022-01-06", "2022-04-15", "2022-05-01",
    "2022-08-15", "2022-10-12", "2022-11-01", "2022-12-06",
    "2022-12-08", "2022-12-25",
    "2023-01-01", "2023-01-06", "2023-04-07", "2023-05-01",
    "2023-08-15", "2023-10-12", "2023-11-01", "2023-12-06",
    "2023-12-08", "2023-12-25",
])

FEATURE_COLUMNS = [
    "dia_semana", "mes", "semana_anio", "trimestre",
    "es_finde", "es_inicio_mes", "es_fin_mes",
    "es_festivo", "dia_post_festivo", "es_black_friday",
    "dias_desde_inicio",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7",  "rolling_std_7",  "rolling_max_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_30", "rolling_std_30",
]
TARGET = "ventas"


def add_calendar_features(df: pd.DataFrame, date_col: str = "fecha") -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["dia_semana"]    = d.dt.dayofweek
    df["mes"]           = d.dt.month
    df["semana_anio"]   = d.dt.isocalendar().week.astype(int)
    df["trimestre"]     = d.dt.quarter
    df["es_finde"]      = (d.dt.dayofweek >= 5).astype(int)
    df["es_inicio_mes"] = (d.dt.day <= 5).astype(int)
    df["es_fin_mes"]    = (d.dt.day >= 25).astype(int)
    df["es_festivo"]    = d.isin(FESTIVOS_ES).astype(int)
    df["dia_post_festivo"] = d.isin(FESTIVOS_ES + pd.Timedelta(days=1)).astype(int)
    df["es_black_friday"] = (
        (d.dt.month == 11) & (d.dt.dayofweek == 4) & (d.dt.day >= 23)
    ).astype(int)
    return df


def add_lag_features(df: pd.DataFrame,
                     lags: List[int] = [1, 7, 14, 21, 28]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame,
                          windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    df = df.copy()
    shifted = df[TARGET].shift(1)   # shift para evitar data leakage
    for w in windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std()
        if w == 7:
            df["rolling_max_7"] = shifted.rolling(w).max()
    return df


def build_feature_matrix(series: pd.Series) -> pd.DataFrame:

    df = series.reset_index()
    df.columns = ["fecha", TARGET]
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Tendencia: días desde el inicio
    df["dias_desde_inicio"] = (df["fecha"] - df["fecha"].min()).dt.days

    # Eliminar filas con NaN generadas por lags/rolling
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: sin mezclar el futuro en el pasado (sin data leakage)."""
    split_idx = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    logger.info(f"Split temporal → Train: {len(train)} | Test: {len(test)}")
    return train, test
