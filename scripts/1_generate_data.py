import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.data_generation import generate_retail_sales, save_data
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    print("=" * 60)
    print("  DataMind Retail — Paso 1: Generación de Datos")
    print("=" * 60)

    df = generate_retail_sales(
        n_stores=cfg.N_STORES,
        n_products=cfg.N_PRODUCTS,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
    )

    filepath = save_data(df, cfg.RAW_DIR)

    print("\nResumen del dataset generado:")
    print(f"  Registros totales : {len(df):>12,}")
    print(f"  Tiendas           : {df['tienda'].nunique():>12}")
    print(f"  Productos         : {df['producto'].nunique():>12}")
    print(f"  Período           : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Venta media/día   : {df['ventas'].mean():>11.1f} uds")
    print(f"\n  Archivo: {filepath}")
    print("\n  Siguiente paso: python scripts/2_spark_processing.py")


if __name__ == "__main__":
    main()
