import os
import subprocess


def _set_java17_if_needed() -> None:
    try:
        version_out = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, text=True
        )
        if 'version "17' in version_out or 'version "1.' in version_out:
            return  # Ya es Java 8/11/17, no hace falta cambiar
        # Intentar localizar Java 17 con la utilidad macOS
        java17 = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if java17:
            os.environ["JAVA_HOME"] = java17
    except Exception:
        pass  # Si falla, dejamos que Spark intente con el Java actual


_set_java17_if_needed()

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType,
)
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Sesión Spark

def create_spark_session(app_name: str = "DataMind_Retail") -> SparkSession:
  
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    logger.info(f"SparkSession creada | versión Spark: {spark.version}")
    return spark


# Carga de datos

SCHEMA = StructType([
    StructField("fecha",       StringType(),  True),
    StructField("tienda",      StringType(),  True),
    StructField("producto",    StringType(),  True),
    StructField("categoria",   StringType(),  True),
    StructField("ventas",      DoubleType(),  True),
    StructField("stock",       IntegerType(), True),
    StructField("es_festivo",  IntegerType(), True),
    StructField("dia_semana",  IntegerType(), True),
    StructField("mes",         IntegerType(), True),
    StructField("semana_anio", IntegerType(), True),
    StructField("trimestre",   IntegerType(), True),
])


def load_sales_data(spark: SparkSession, filepath: str) -> DataFrame:
    df = (
        spark.read
        .option("header", "true")
        .schema(SCHEMA)
        .csv(filepath)
        .withColumn("fecha", F.to_date("fecha", "yyyy-MM-dd"))
    )
    logger.info(f"Spark: {df.count():,} registros cargados")
    return df


# Transformaciones

def compute_daily_store_aggregates(df: DataFrame) -> DataFrame:
    """Agrega ventas, productos únicos y stock por tienda y día."""
    return (
        df.groupBy("fecha", "tienda")
        .agg(
            F.sum("ventas").alias("ventas_total"),
            F.countDistinct("producto").alias("productos_distintos"),
            F.avg("ventas").alias("venta_media_producto"),
            F.sum("stock").alias("stock_total"),
            F.max("es_festivo").alias("es_festivo"),
        )
        .orderBy("tienda", "fecha")
    )


def compute_category_weekly_stats(df: DataFrame) -> DataFrame:
    """Estadísticas de ventas semanales agrupadas por categoría."""
    return (
        df.groupBy(
            F.year("fecha").alias("anio"),
            F.weekofyear("fecha").alias("semana"),
            "categoria",
        )
        .agg(
            F.sum("ventas").alias("ventas_semana"),
            F.avg("ventas").alias("venta_media"),
            F.stddev("ventas").alias("desviacion"),
            F.count("*").alias("num_registros"),
        )
        .orderBy("anio", "semana", "categoria")
    )


def compute_rolling_avg_spark(df: DataFrame, window_days: int = 7) -> DataFrame:
    """
    Media móvil usando Window Functions de Spark.
    Demuestra el uso de funciones analíticas sobre datos distribuidos.
    """
    secs = window_days * 86_400
    w = (
        Window
        .partitionBy("tienda", "producto")
        .orderBy(F.col("fecha").cast("timestamp").cast("long"))
        .rangeBetween(-secs, 0)
    )
    return df.withColumn(f"rolling_mean_{window_days}d", F.avg("ventas").over(w))


def detect_stock_alerts(df: DataFrame,
                         alert_ratio: float = 0.3) -> DataFrame:
    """
    Clasifica el estado de stock de cada producto/tienda:
      - CRITICO: stock < 30 % de la venta media de los últimos 7 días
      - BAJO:    stock < venta media 7 días
      - OK:      resto
    """
    w7 = (
        Window
        .partitionBy("tienda", "producto")
        .orderBy("fecha")
        .rowsBetween(-7, -1)
    )
    return (
        df
        .withColumn("avg_ventas_7d", F.avg("ventas").over(w7))
        .withColumn(
            "alerta_stock",
            F.when(F.col("stock") < F.col("avg_ventas_7d") * alert_ratio, "CRITICO")
             .when(F.col("stock") < F.col("avg_ventas_7d"), "BAJO")
             .otherwise("OK"),
        )
    )


def top_products_by_store(df: DataFrame, top_n: int = 5) -> DataFrame:
    """Ranking de los N productos con más ventas en cada tienda."""
    w_rank = Window.partitionBy("tienda").orderBy(F.desc("total_ventas"))
    return (
        df.groupBy("tienda", "producto", "categoria")
        .agg(F.sum("ventas").alias("total_ventas"))
        .withColumn("ranking", F.rank().over(w_rank))
        .filter(F.col("ranking") <= top_n)
        .orderBy("tienda", "ranking")
    )


# Pipeline completo

def run_full_pipeline(input_csv: str, output_dir: str) -> None:
    """
    Ejecuta el pipeline completo de Batch Processing con Spark.

    Genera cuatro ficheros CSV en output_dir:
      - daily_store_aggregates.csv
      - weekly_category_stats.csv
      - top_products_by_store.csv
      - stock_alerts.csv
    """
    spark = create_spark_session()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=== Batch Layer (Apache Spark) — inicio ===")

    df = load_sales_data(spark, input_csv)
    df.cache()

    logger.info("1/4 Agregados diarios por tienda...")
    compute_daily_store_aggregates(df).toPandas().to_csv(
        out / "daily_store_aggregates.csv", index=False
    )

    logger.info("2/4 Estadísticas semanales por categoría...")
    compute_category_weekly_stats(df).toPandas().to_csv(
        out / "weekly_category_stats.csv", index=False
    )

    logger.info("3/4 Top productos por tienda...")
    top_products_by_store(df, top_n=5).toPandas().to_csv(
        out / "top_products_by_store.csv", index=False
    )

    logger.info("4/4 Alertas de stock...")
    alerts = detect_stock_alerts(df)
    critical = (
        alerts
        .filter(F.col("alerta_stock").isin("CRITICO", "BAJO"))
        .select("fecha", "tienda", "producto", "stock", "avg_ventas_7d", "alerta_stock")
        .orderBy(F.desc("fecha"))
    )
    critical.toPandas().to_csv(out / "stock_alerts.csv", index=False)

    n_alerts = critical.filter(F.col("alerta_stock") == "CRITICO").count()
    logger.info(f"=== Pipeline completado | Alertas críticas: {n_alerts} ===")

    spark.stop()
