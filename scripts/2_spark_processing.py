import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import config as cfg
from src.bigdata.spark_processor import run_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    print("=" * 60)
    print("  DataMind Retail — Paso 2: Batch Layer con Apache Spark")
    print("=" * 60)

    input_csv = str(cfg.RAW_DIR / "ventas_raw.csv")
    output_dir = str(cfg.PROCESSED_DIR)

    if not Path(input_csv).exists():
        print("ERROR: Ejecuta primero el script 1 para generar los datos.")
        sys.exit(1)

    run_full_pipeline(input_csv, output_dir)

    print("\nFicheros generados en data/processed/:")
    for f in sorted(Path(output_dir).glob("*.csv")):
        rows = sum(1 for _ in open(f)) - 1
        print(f"  {f.name:<40} {rows:>8,} filas")

    print("\n  Siguiente paso: python scripts/3_train_models.py")


if __name__ == "__main__":
    main()
