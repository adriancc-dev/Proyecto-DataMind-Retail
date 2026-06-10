import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class RandomForestDemandModel:

    name = "RandomForest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 15,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            feature_columns: List[str]) -> "RandomForestDemandModel":
        self.feature_columns = feature_columns
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=self.random_state,
            )),
        ])
        self.pipeline.fit(X_train[feature_columns], y_train)
        top5 = self.get_feature_importance().head(5).to_dict()
        logger.info(f"[{self.name}] Entrenado | Top features: {top5}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.pipeline.predict(X[self.feature_columns])
        return np.maximum(0.0, preds)

    def get_feature_importance(self) -> pd.Series:
        rf = self.pipeline.named_steps["rf"]
        return (
            pd.Series(rf.feature_importances_, index=self.feature_columns)
            .sort_values(ascending=False)
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / f"{self.name}.pkl")
        logger.info(f"[{self.name}] Guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> "RandomForestDemandModel":
        return joblib.load(path / f"{cls.name}.pkl")


class GradientBoostingDemandModel(RandomForestDemandModel):

    name = "GradientBoosting"

    def __init__(self, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            feature_columns: List[str]) -> "GradientBoostingDemandModel":
        self.feature_columns = feature_columns
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=5,
                random_state=self.random_state,
            )),
        ])
        self.pipeline.fit(X_train[feature_columns], y_train)
        logger.info(f"[{self.name}] Entrenado")
        return self

    def get_feature_importance(self) -> pd.Series:
        gb = self.pipeline.named_steps["gb"]
        return (
            pd.Series(gb.feature_importances_, index=self.feature_columns)
            .sort_values(ascending=False)
        )
