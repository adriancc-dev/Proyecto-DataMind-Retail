import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

logger = logging.getLogger(__name__)


class MovingAverageModel:

    def __init__(self, window: int = 7):
        self.window = window
        self.name = f"MediaMovil_{window}d"
        self._mean: float = 0.0

    def fit(self, series: pd.Series) -> "MovingAverageModel":
        self._mean = float(series.values[-self.window:].mean())
        logger.info(f"[{self.name}] Ajustado | media={self._mean:.2f}")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._mean)


class HoltWintersModel:

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.name = "HoltWinters"
        self._fitted = None

    def fit(self, series: pd.Series) -> "HoltWintersModel":
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self._fitted = model.fit(optimized=True, use_brute=False)
        logger.info(f"[{self.name}] Ajustado | AIC={self._fitted.aic:.2f}")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Llama a fit() antes de predict().")
        forecast = self._fitted.forecast(steps=horizon).values
        return np.maximum(0.0, forecast)
