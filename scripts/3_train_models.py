import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import mlflow
import config as cfg
from src.preprocessing import load_raw_data, clean_data, get_product_series
from src.features import (
    build_feature_matrix, train_test_split_temporal, FEATURE_COLUMNS,
)
from src.models.baseline import MovingAverageModel, HoltWintersModel
from src.models.random_forest import RandomForestDemandModel
from src.models.lstm import LSTMDemandModel
from src.evaluation import evaluate_model, compare_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_training(tienda: str, producto: str) -> pd.DataFrame:
    # ── Cargar y preparar datos ───────────────────────────────────────────────
    df = clean_data(load_raw_data(cfg.RAW_DIR / "ventas_raw.csv"))
    series = get_product_series(df, tienda, producto)

    split_idx = int(len(series) * (1 - cfg.TEST_SIZE))
    s_train = series.iloc[:split_idx]
    s_test  = series.iloc[split_idx:]

    print(f"\n  Tienda   : {tienda}")
    print(f"  Producto : {producto}")
    print(f"  Train    : {s_train.index[0].date()} → {s_train.index[-1].date()} "
          f"({len(s_train)} días)")
    print(f"  Test     : {s_test.index[0].date()} → {s_test.index[-1].date()} "
          f"({len(s_test)} días)\n")

    # Preparar features para modelos de ML
    full_series = pd.concat([s_train, s_test])
    feat_df = build_feature_matrix(full_series)
    train_feat, test_feat = train_test_split_temporal(feat_df)
    avail_features = [c for c in FEATURE_COLUMNS if c in train_feat.columns]

    results = []
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Configurar MLflow
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{tienda}_{producto}"):
        mlflow.log_params({
            "tienda": tienda, "producto": producto,
            "train_dias": len(s_train), "test_dias": len(s_test),
            "lstm_lookback": cfg.LSTM_LOOKBACK, "lstm_epochs": cfg.LSTM_EPOCHS,
        })

        # 1. Media Móvil
        print("  [1/4] Media Móvil 7 días...")
        ma = MovingAverageModel(window=7)
        ma.fit(s_train)
        y_pred_ma = ma.predict(len(s_test))
        res_ma = evaluate_model(s_test.values, y_pred_ma, ma.name)
        results.append(res_ma)
        mlflow.log_metrics({f"{ma.name}_MAPE": res_ma["MAPE (%)"],
                             f"{ma.name}_MAE":  res_ma["MAE"],
                             f"{ma.name}_RMSE": res_ma["RMSE"]})

        # 2. Holt-Winters
        print("  [2/4] Holt-Winters...")
        hw = HoltWintersModel(seasonal_periods=7)
        hw.fit(s_train)
        y_pred_hw = hw.predict(len(s_test))
        res_hw = evaluate_model(s_test.values, y_pred_hw, hw.name)
        results.append(res_hw)
        mlflow.log_metrics({f"{hw.name}_MAPE": res_hw["MAPE (%)"],
                             f"{hw.name}_MAE":  res_hw["MAE"],
                             f"{hw.name}_RMSE": res_hw["RMSE"]})

        # 3. Random Forest
        print("  [3/4] Random Forest...")
        rf = RandomForestDemandModel(n_estimators=200)
        rf.fit(train_feat, train_feat["ventas"], avail_features)
        y_pred_rf = rf.predict(test_feat)
        res_rf = evaluate_model(test_feat["ventas"].values, y_pred_rf, rf.name)
        results.append(res_rf)
        rf.save(cfg.MODELS_DIR)
        mlflow.log_metrics({f"{rf.name}_MAPE": res_rf["MAPE (%)"],
                             f"{rf.name}_MAE":  res_rf["MAE"],
                             f"{rf.name}_RMSE": res_rf["RMSE"]})

        # 4. LSTM
        print("  [4/4] LSTM (TensorFlow/Keras)...")
        lstm = LSTMDemandModel(
            lookback=cfg.LSTM_LOOKBACK,
            epochs=cfg.LSTM_EPOCHS,
            batch_size=cfg.LSTM_BATCH_SIZE,
        )
        lstm.fit(s_train)
        y_pred_lstm = lstm.predict_test(s_train, s_test)
        res_lstm = evaluate_model(s_test.values, y_pred_lstm, lstm.name)
        results.append(res_lstm)
        lstm.save(cfg.MODELS_DIR)
        mlflow.log_metrics({f"{lstm.name}_MAPE": res_lstm["MAPE (%)"],
                             f"{lstm.name}_MAE":  res_lstm["MAE"],
                             f"{lstm.name}_RMSE": res_lstm["RMSE"]})

    return compare_models(results)


def main():
    print("=" * 60)
    print("  DataMind Retail — Paso 3: Entrenamiento de Modelos")
    print("=" * 60)

    if not (cfg.RAW_DIR / "ventas_raw.csv").exists():
        print("ERROR: Ejecuta primero el script 1.")
        sys.exit(1)

    comparison = run_training(cfg.DEFAULT_STORE, cfg.DEFAULT_PRODUCT)

    print("\n" + "=" * 60)
    print("  COMPARATIVA DE MODELOS (Test set)")
    print("=" * 60)
    print(comparison.to_string())

    best = comparison.index[0]
    best_mape = comparison.iloc[0]["MAPE (%)"]
    objetivo = "✅ CUMPLIDO" if best_mape < 15 else "⚠️  Pendiente optimización"
    print(f"\n  Mejor modelo : {best}")
    print(f"  MAPE         : {best_mape:.2f}%  ← Objetivo < 15 %  {objetivo}")

    comparison.to_csv(cfg.MODELS_DIR / "model_comparison.csv")
    print(f"\n  Resultados guardados en {cfg.MODELS_DIR / 'model_comparison.csv'}")
    print(f"  MLflow UI: ejecuta 'mlflow ui' y abre http://localhost:5000")
    print("\n  Siguiente paso: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
