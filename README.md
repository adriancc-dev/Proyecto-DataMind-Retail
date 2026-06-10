# 🛒 DataMind Retail — Predicción de demanda y gestión de inventario

Proyecto final del **Curso de Especialización en Inteligencia Artificial y Big Data**.

Pipeline completo de ciencia de datos para retail: genera datos de ventas sintéticos,
los procesa con **Apache Spark**, entrena y compara **4 modelos de predicción de demanda**
(clásicos, machine learning y deep learning), registra los experimentos con **MLflow** y
expone los resultados en un **dashboard interactivo de Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.4+-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B)

---

## 🎯 Objetivo

Predecir la demanda diaria de productos por tienda para optimizar el inventario y
generar alertas de stock, demostrando un flujo de trabajo end-to-end que cubre las
tecnologías clave de un perfil de **Data Scientist / Data Engineer**.

## 🏗️ Arquitectura

Inspirado en la **arquitectura Lambda** (batch + serving layer):

```
┌────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌───────────────┐
│ 1. Generación  │   │ 2. Batch Layer  │   │ 3. Modelado +    │   │ 4. Serving     │
│    de datos    │──▶│   Apache Spark  │──▶│    MLflow        │──▶│   Streamlit    │
│  (datos raw)   │   │  (agregaciones) │   │  (4 modelos)     │   │  (dashboard)   │
└────────────────┘   └─────────────────┘   └──────────────────┘   └───────────────┘
```

## 🧰 Stack tecnológico

| Capa              | Tecnologías                                         |
|-------------------|-----------------------------------------------------|
| Procesamiento     | **PySpark** (agregaciones batch), Pandas, NumPy      |
| Machine Learning  | **scikit-learn** (Random Forest), statsmodels        |
| Deep Learning     | **TensorFlow / Keras** (LSTM)                        |
| MLOps             | **MLflow** (tracking de experimentos y métricas)     |
| Visualización     | **Streamlit**, Plotly                                |
| Ingeniería de datos | Feature engineering temporal (lags, rolling, calendario español, Black Friday) |

## 🤖 Modelos comparados

Se entrenan y evalúan cuatro enfoques sobre la misma serie temporal:

1. **Media Móvil 7d** — baseline ingenuo
2. **Holt-Winters** — suavizado exponencial con estacionalidad
3. **Random Forest** — ML con +20 features temporales
4. **LSTM** — red neuronal recurrente para secuencias

### 📊 Resultados

| Modelo          | Ranking | MAPE (%) | MAE  | RMSE |
|-----------------|:-------:|:--------:|:----:|:----:|
| **RandomForest**|  🥇 1   | **13.89**| 3.61 | 6.45 |
| LSTM            |   2     |  24.50   | 5.91 | 8.93 |
| HoltWinters     |   3     |  24.84   | 6.81 | 9.36 |
| MediaMovil 7d   |   4     |  32.27   | 9.92 | 13.20|

> El **Random Forest** con feature engineering temporal obtiene el mejor MAPE (13.89%),
> superando incluso al LSTM en este dataset.

## 📁 Estructura del proyecto

```
codigo/
├── config.py                 # Configuración global (rutas, parámetros)
├── requirements.txt
├── scripts/                  # Pipeline ejecutable paso a paso
│   ├── 1_generate_data.py    #   → genera ventas_raw.csv
│   ├── 2_spark_processing.py #   → batch layer con Spark
│   └── 3_train_models.py     #   → entrena y registra modelos en MLflow
├── src/
│   ├── data_generation.py    # Generación de datos sintéticos
│   ├── preprocessing.py      # Limpieza y series por producto
│   ├── features.py           # Feature engineering temporal
│   ├── evaluation.py         # Métricas (MAPE, MAE, RMSE)
│   ├── bigdata/
│   │   └── spark_processor.py# Agregaciones con PySpark
│   └── models/
│       ├── baseline.py       # Media móvil + Holt-Winters
│       ├── random_forest.py  # Random Forest
│       └── lstm.py           # LSTM (Keras)
└── dashboard/
    └── app.py                # Dashboard Streamlit
```

> `data/`, `models_saved/` y `mlruns/` se generan al ejecutar el pipeline y están
> excluidos del repositorio (ver `.gitignore`).

## 🚀 Cómo ejecutarlo

### 1. Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **TensorFlow:** en `requirements.txt`, descomenta la línea que corresponda a tu
> sistema (`tensorflow-macos` para Apple Silicon, `tensorflow` para Windows/Linux/Intel).

### 2. Ejecutar el pipeline

```bash
python scripts/1_generate_data.py     # Genera los datos de ventas
python scripts/2_spark_processing.py  # Procesa con Spark (batch layer)
python scripts/3_train_models.py      # Entrena y compara los 4 modelos
```

### 3. Visualizar resultados

```bash
# Dashboard interactivo
streamlit run dashboard/app.py

# Experimentos MLflow
mlflow ui --backend-store-uri mlruns
```

---

## 👤 Autor

**Adrián Cabedo Cañós** — Curso de Especialización en IA y Big Data.

> Proyecto desarrollado con fines formativos y de portfolio.
