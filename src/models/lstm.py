import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

tf.random.set_seed(42)
np.random.seed(42)


def _create_sequences(values: np.ndarray,
                       lookback: int) -> tuple[np.ndarray, np.ndarray]:
 
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback: i])
        y.append(values[i])
    return np.array(X)[..., np.newaxis], np.array(y)


class LSTMDemandModel:

    name = "LSTM"

    def __init__(self, lookback: int = 30, epochs: int = 50,
                 batch_size: int = 32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self._last_sequence: Optional[np.ndarray] = None

    # ── Arquitectura ─────────────────────────────────────────────────────────
    def _build_model(self) -> Sequential:
        m = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ], name="lstm_demand")
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="huber",
            metrics=["mae"],
        )
        return m

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    def fit(self, series: pd.Series,
            validation_split: float = 0.15) -> "LSTMDemandModel":
    
        values = series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values).flatten()

        X, y = _create_sequences(scaled, self.lookback)

        # Split temporal para validación (respeta el orden cronológico)
        val_size = max(1, int(len(X) * validation_split))
        X_tr, X_val = X[:-val_size], X[-val_size:]
        y_tr, y_val = y[:-val_size], y[-val_size:]

        self.model = self._build_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0),
        ]

        logger.info(
            f"[{self.name}] Entrenando | lookback={self.lookback} | "
            f"max_epochs={self.epochs} | train={len(X_tr)} | val={len(X_val)}"
        )

        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        real_epochs = len(self.history.history["loss"])
        best_loss = min(self.history.history["val_loss"])
        logger.info(
            f"[{self.name}] Completado | epochs={real_epochs} | "
            f"best_val_loss={best_loss:.4f}"
        )

        self._last_sequence = scaled[-self.lookback:].copy()
        return self

    # Predicción sobre test
    def predict_test(self, series_train: pd.Series,
                     series_test: pd.Series) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de predict_test().")

        full = np.concatenate([series_train.values, series_test.values]).reshape(-1, 1)
        scaled = self.scaler.transform(full).flatten()

        start = len(series_train)
        preds_scaled = []
        for i in range(len(series_test)):
            window = scaled[start + i - self.lookback: start + i]
            X = window.reshape(1, self.lookback, 1)
            preds_scaled.append(float(self.model.predict(X, verbose=0)[0, 0]))

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(0.0, preds)

    # Predicción futura
    def forecast(self, horizon: int) -> np.ndarray:
        if self.model is None or self._last_sequence is None:
            raise RuntimeError("Llama a fit() antes de forecast().")

        seq = self._last_sequence.copy()
        preds_scaled = []
        for _ in range(horizon):
            X = seq[-self.lookback:].reshape(1, self.lookback, 1)
            p = float(self.model.predict(X, verbose=0)[0, 0])
            preds_scaled.append(p)
            seq = np.append(seq, p)

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(0.0, preds)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history.history) if self.history else pd.DataFrame()

    # ── Persistencia ─────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / f"{self.name}_model.keras")
        joblib.dump(self.scaler, path / f"{self.name}_scaler.pkl")
        np.save(path / f"{self.name}_last_seq.npy", self._last_sequence)
        logger.info(f"[{self.name}] Guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> "LSTMDemandModel":
        from tensorflow.keras.models import load_model
        obj = cls()
        obj.model = load_model(path / f"{cls.name}_model.keras")
        obj.scaler = joblib.load(path / f"{cls.name}_scaler.pkl")
        obj._last_sequence = np.load(path / f"{cls.name}_last_seq.npy")
        return obj
