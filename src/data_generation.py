import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Categorías por producto
PRODUCT_CATEGORIES = {
    "PROD_001": "Alimentación", "PROD_002": "Alimentación",
    "PROD_003": "Alimentación", "PROD_004": "Alimentación",
    "PROD_005": "Moda",         "PROD_006": "Moda",
    "PROD_007": "Moda",         "PROD_008": "Moda",
    "PROD_009": "Moda",         "PROD_010": "Hogar",
    "PROD_011": "Hogar",        "PROD_012": "Hogar",
    "PROD_013": "Hogar",        "PROD_014": "Deportes",
    "PROD_015": "Deportes",     "PROD_016": "Deportes",
    "PROD_017": "Deportes",     "PROD_018": "Electrónica",
    "PROD_019": "Electrónica",  "PROD_020": "Electrónica",
}

# Festivos nacionales España 2022-2023
_FESTIVOS_LIST = [
    "2022-01-01", "2022-01-06", "2022-04-15", "2022-05-01",
    "2022-08-15", "2022-10-12", "2022-11-01", "2022-12-06",
    "2022-12-08", "2022-12-25",
    "2023-01-01", "2023-01-06", "2023-04-07", "2023-05-01",
    "2023-08-15", "2023-10-12", "2023-11-01", "2023-12-06",
    "2023-12-08", "2023-12-25",
]
# Usamos un set de date() para búsqueda O(1)
FESTIVOS_SET = set(pd.to_datetime(_FESTIVOS_LIST).date)

# Factor de estacionalidad semanal: clave = día de semana (0=lun, 6=dom)
WEEKLY_FACTOR = {0: 0.85, 1: 0.88, 2: 0.92, 3: 0.95, 4: 1.20, 5: 1.35, 6: 0.85}

# Factor de estacionalidad mensual
MONTHLY_FACTOR = {
    1: 0.90, 2: 0.85,  3: 0.95, 4: 0.98,  5: 1.00, 6: 1.05,
    7: 1.10, 8: 0.80,  9: 1.00, 10: 1.05, 11: 1.20, 12: 1.40,
}


def generate_retail_sales(
    n_stores: int = 5,
    n_products: int = 20,
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    random_seed: int = 42,
) -> pd.DataFrame:
 
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    t0 = pd.Timestamp(start_date)

    stores = [f"TIENDA_{i:02d}" for i in range(1, n_stores + 1)]
    products = [f"PROD_{i:03d}" for i in range(1, n_products + 1)]

    records = []
    for store in stores:
        store_factor = rng.uniform(0.8, 1.2)  # cada tienda tiene distinto volumen
        for product in products:
            prod_num = int(product.split("_")[1])
            base_demand = 20.0 + prod_num * 4.5  # rango ~24–110 uds/día

            for date in dates:
                # Tendencia creciente suave (2 % anual)
                days_elapsed = (date - t0).days
                trend = 1.0 + 0.02 * (days_elapsed / 365.0)

                # Estacionalidades
                w_factor = WEEKLY_FACTOR[date.dayofweek]
                m_factor = MONTHLY_FACTOR[date.month]

                # Festivos → caída fuerte de ventas
                is_festivo = 1 if date.date() in FESTIVOS_SET else 0
                festivo_factor = 0.30 if is_festivo else 1.0

                # Black Friday: cuarto viernes de noviembre
                is_black_friday = (
                    date.month == 11
                    and date.dayofweek == 4
                    and date.day >= 23
                )
                bf_factor = 2.5 if is_black_friday else 1.0

                demand = (
                    base_demand
                    * trend
                    * w_factor
                    * m_factor
                    * festivo_factor
                    * bf_factor
                    * store_factor
                    * rng.uniform(0.85, 1.15)  # ruido diario ±15 %
                )
                ventas = max(0, int(round(demand)))
                # Stock entre 1.5× y 4× la venta del día (simplificación)
                stock = max(0, int(round(demand * rng.uniform(1.5, 4.0))))

                records.append({
                    "fecha":       date.strftime("%Y-%m-%d"),
                    "tienda":      store,
                    "producto":    product,
                    "categoria":   PRODUCT_CATEGORIES.get(product, "Otros"),
                    "ventas":      float(ventas),
                    "stock":       stock,
                    "es_festivo":  is_festivo,
                    "dia_semana":  date.dayofweek,
                    "mes":         date.month,
                    "semana_anio": date.isocalendar()[1],
                    "trimestre":   (date.month - 1) // 3 + 1,
                })

    df = pd.DataFrame(records)
    # Convertir fecha a datetime para uso downstream
    df["fecha"] = pd.to_datetime(df["fecha"])
    logger.info(
        f"Dataset generado: {len(df):,} registros | "
        f"{n_stores} tiendas × {n_products} productos × {len(dates)} días"
    )
    return df


def save_data(df: pd.DataFrame, directory: Path) -> Path:
    """Guarda el DataFrame como CSV. Retorna la ruta del archivo."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "ventas_raw.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"Datos guardados en {filepath}")
    return filepath
