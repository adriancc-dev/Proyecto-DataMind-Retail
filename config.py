"""
Configuración global del proyecto DataMind Retail.
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models_saved"

# Parámetros de generación de datos
N_STORES = 5
N_PRODUCTS = 20
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# Parámetros de los modelos
TEST_SIZE = 0.2
LSTM_LOOKBACK = 30
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# MLflow
MLFLOW_TRACKING_URI = str(BASE_DIR / "mlruns")
EXPERIMENT_NAME = "datamind_retail"

# Producto y tienda objetivo por defecto para los scripts
DEFAULT_STORE = "TIENDA_01"
DEFAULT_PRODUCT = "PROD_001"
