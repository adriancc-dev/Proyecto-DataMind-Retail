import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_raw_data(filepath) -> pd.DataFrame:

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {filepath}\n"
            "Ejecuta primero: python scripts/1_generate_data.py"
        )
    df = pd.read_csv(filepath, parse_dates=["fecha"])
    logger.info(f"Datos cargados: {len(df):,} registros desde {filepath.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
 
    original_len = len(df)

    df = df.drop_duplicates(subset=["fecha", "tienda", "producto"])

    df["ventas"] = pd.to_numeric(df["ventas"], errors="coerce").fillna(0).clip(lower=0)
    df["stock"]  = pd.to_numeric(df["stock"],  errors="coerce").fillna(0).clip(lower=0)
    df["fecha"]  = pd.to_datetime(df["fecha"])

    df = df.sort_values(["tienda", "producto", "fecha"]).reset_index(drop=True)

    dropped = original_len - len(df)
    if dropped > 0:
        logger.warning(f"Limpieza: {dropped} duplicados eliminados")

    logger.info(f"Datos limpios: {len(df):,} registros")
    return df


def get_product_series(df: pd.DataFrame, tienda: str, producto: str) -> pd.Series:
   
    mask = (df["tienda"] == tienda) & (df["producto"] == producto)
    subset = df.loc[mask, ["fecha", "ventas"]].set_index("fecha")["ventas"].sort_index()

    if len(subset) == 0:
        raise ValueError(
            f"No hay datos para tienda='{tienda}', producto='{producto}'"
        )

    # Rellenar huecos en el índice con 0
    full_idx = pd.date_range(subset.index.min(), subset.index.max(), freq="D")
    subset = subset.reindex(full_idx, fill_value=0.0)
    subset.name = "ventas"
    return subset
