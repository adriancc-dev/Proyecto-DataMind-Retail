"""
DataMind Retail — Dashboard Streamlit
Visualización interactiva de ventas, predicciones e inventario.
Ejecutar: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

from src.preprocessing import load_raw_data, clean_data, get_product_series
from src.features import (
    build_feature_matrix, train_test_split_temporal, FEATURE_COLUMNS,
)
from src.models.baseline import MovingAverageModel, HoltWintersModel
from src.models.random_forest import RandomForestDemandModel
from src.models.lstm import LSTMDemandModel
from src.evaluation import evaluate_model, compare_models
import config as cfg

# Configuración de página
st.set_page_config(
    page_title="DataMind Retail",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# Carga de datos
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    fp = cfg.RAW_DIR / "ventas_raw.csv"
    if not fp.exists():
        return None
    return clean_data(load_raw_data(fp))


@st.cache_data
def load_processed(filename: str) -> Optional[pd.DataFrame]:
    fp = cfg.PROCESSED_DIR / filename
    return pd.read_csv(fp) if fp.exists() else None


# Entrenamiento cacheado (evita re-entrenar al cambiar parámetros)
@st.cache_resource(show_spinner=False)
def train_all_models(tienda: str, producto: str):
    """
    Entrena los 4 modelos y devuelve predicciones + métricas.
    El resultado queda cacheado por Streamlit entre interacciones.
    """
    df = load_data()
    series = get_product_series(df, tienda, producto)

    split = int(len(series) * 0.8)
    s_tr, s_te = series.iloc[:split], series.iloc[split:]

    full = pd.concat([s_tr, s_te])
    feat = build_feature_matrix(full)
    tr_f, te_f = train_test_split_temporal(feat)
    avail = [c for c in FEATURE_COLUMNS if c in tr_f.columns]

    preds, results = {}, []

    # Media Móvil
    ma = MovingAverageModel(window=7)
    ma.fit(s_tr)
    p_ma = ma.predict(len(s_te))
    preds["Media Móvil 7d"] = (s_te.index, p_ma)
    results.append(evaluate_model(s_te.values, p_ma, ma.name))

    # Holt-Winters
    hw = HoltWintersModel()
    hw.fit(s_tr)
    p_hw = hw.predict(len(s_te))
    preds["Holt-Winters"] = (s_te.index, p_hw)
    results.append(evaluate_model(s_te.values, p_hw, hw.name))

    # Random Forest
    rf = RandomForestDemandModel(n_estimators=100)
    rf.fit(tr_f, tr_f["ventas"], avail)
    p_rf = rf.predict(te_f)
    preds["Random Forest"] = (te_f["fecha"] if "fecha" in te_f else s_te.index, p_rf)
    results.append(evaluate_model(te_f["ventas"].values, p_rf, rf.name))

    # LSTM (epochs reducidos para el dashboard)
    lstm = LSTMDemandModel(lookback=30, epochs=25, batch_size=32)
    lstm.fit(s_tr, validation_split=0.15)
    p_lstm = lstm.predict_test(s_tr, s_te)
    preds["LSTM"] = (s_te.index, p_lstm)
    results.append(evaluate_model(s_te.values, p_lstm, lstm.name))

    return series, s_tr, s_te, preds, results, te_f, rf


# SIDEBAR
st.sidebar.title("DataMind Retail")
st.sidebar.caption("Sistema de Predicción de Demanda con IA y Big Data")
st.sidebar.markdown("---")

df = load_data()

if df is None:
    st.error(
        "No se encontraron datos. "
        "Ejecuta: `python scripts/1_generate_data.py`"
    )
    st.stop()

tiendas  = sorted(df["tienda"].unique())
products = sorted(df["producto"].unique())

sel_store   = st.sidebar.selectbox("Tienda", tiendas)
sel_product = st.sidebar.selectbox("Producto", products)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset cargado**  \n"
    f"- `{len(df):,}` registros  \n"
    f"- `{df['tienda'].nunique()}` tiendas  \n"
    f"- `{df['producto'].nunique()}` productos  \n"
    f"- `{df['fecha'].min().date()}` → `{df['fecha'].max().date()}`"
)


# CABECERA
st.title("📊 DataMind Retail Analytics")
st.markdown(
    "*Sistema de Predicción de Demanda y Optimización de Inventario  "
    "— Proyecto Final IA & Big Data 2025/26*"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Visión General",
    "🔮 Predicciones de Demanda",
    "📦 Inventario y Alertas",
    "🏆 Comparativa de Modelos",
])


# TAB 1 — VISIÓN GENERAL
with tab1:
    st.subheader(f"Resumen Ejecutivo — {sel_store}")

    store_df = df[df["tienda"] == sel_store]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas Totales (2 años)",  f"{store_df['ventas'].sum():,.0f} uds")
    col2.metric("Media Diaria",
                f"{store_df.groupby('fecha')['ventas'].sum().mean():,.0f} uds/día")
    col3.metric("Productos Activos", str(store_df["producto"].nunique()))
    col4.metric("Stock Medio", f"{store_df['stock'].mean():,.0f} uds")

    st.markdown("---")

    # Ventas diarias totales con media móvil
    daily = (
        store_df.groupby("fecha")["ventas"].sum()
        .reset_index(name="ventas")
    )
    daily["mm7"] = daily["ventas"].rolling(7).mean()

    fig = go.Figure()
    fig.add_bar(x=daily["fecha"], y=daily["ventas"],
                name="Ventas diarias", marker_color="#93c6e7", opacity=0.55)
    fig.add_scatter(x=daily["fecha"], y=daily["mm7"],
                    name="Media móvil 7d", line=dict(color="#1a4a7a", width=2))
    fig.update_layout(
        title=f"Ventas Diarias Totales — {sel_store}",
        xaxis_title="Fecha", yaxis_title="Unidades",
        hovermode="x unified", height=370,
        legend=dict(orientation="h", y=-0.22),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        cat_v = (
            store_df.groupby("categoria")["ventas"].sum()
            .sort_values(ascending=False).reset_index()
        )
        fig2 = px.bar(cat_v, x="categoria", y="ventas",
                      title="Ventas por Categoría",
                      color="ventas", color_continuous_scale="Blues",
                      labels={"ventas": "Uds vendidas", "categoria": "Categoría"})
        fig2.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        dow_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        dow_v = store_df.groupby("dia_semana")["ventas"].mean().reset_index()
        dow_v["día"] = dow_v["dia_semana"].map(dict(enumerate(dow_names)))
        fig3 = px.bar(dow_v, x="día", y="ventas",
                      title="Estacionalidad Semanal",
                      color="ventas", color_continuous_scale="Blues",
                      labels={"ventas": "Ventas medias", "día": "Día"})
        fig3.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# TAB 2 — PREDICCIONES
with tab2:
    st.subheader(f"Predicción de Demanda — {sel_store} | {sel_product}")

    with st.spinner("Entrenando modelos de IA… (primera vez puede tardar ~2 min)"):
        series, s_tr, s_te, preds, results, te_f, rf = train_all_models(
            sel_store, sel_product
        )

    model_opts = list(preds.keys())
    sel_models = st.multiselect(
        "Modelos a visualizar",
        options=model_opts,
        default=["Random Forest", "LSTM"],
    )

    COLORS = {
        "Media Móvil 7d": "#aaaaaa",
        "Holt-Winters":   "#f39c12",
        "Random Forest":  "#2ecc71",
        "LSTM":           "#e74c3c",
    }

    fig_p = go.Figure()
    fig_p.add_scatter(x=s_tr.index, y=s_tr.values,
                      name="Real (train)", line=dict(color="#1a4a7a", width=1),
                      opacity=0.4)
    fig_p.add_scatter(x=s_te.index, y=s_te.values,
                      name="Real (test)", line=dict(color="#1a4a7a", width=2))

    for m in sel_models:
        if m in preds:
            x_p, y_p = preds[m]
            fig_p.add_scatter(
                x=x_p, y=y_p,
                name=f"Pred. {m}",
                line=dict(color=COLORS.get(m, "#888"), width=2, dash="dot"),
            )

    fig_p.add_vline(x=s_te.index[0].timestamp() * 1000,
                    line_dash="dash", line_color="gray",
                    annotation_text="Inicio test")
    fig_p.update_layout(
        title=f"Predicción vs. Real — {sel_product} en {sel_store}",
        xaxis_title="Fecha", yaxis_title="Unidades vendidas",
        hovermode="x unified", height=440,
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    st.subheader("Métricas de error en el período de test")
    comp_df = compare_models(results)

    c1, c2, c3 = st.columns([1, 1, 3])
    best_name = comp_df.index[0]
    best_mape = comp_df.iloc[0]["MAPE (%)"]
    c1.metric("Mejor modelo", best_name)
    c2.metric("Mejor MAPE", f"{best_mape:.1f} %",
               delta="Objetivo < 15 %",
               delta_color="normal" if best_mape < 15 else "inverse")
    with c3:
        st.dataframe(
            comp_df[["MAPE (%)", "MAE", "RMSE"]]
            .style.format("{:.2f}")
            .highlight_min(color="#d4edda")
            .highlight_max(color="#f8d7da"),
            use_container_width=True,
        )


# TAB 3 — INVENTARIO Y ALERTAS
with tab3:
    st.subheader("Estado del Inventario")

    alerts_df = load_processed("stock_alerts.csv")
    if alerts_df is not None:
        n_crit = (alerts_df["alerta_stock"] == "CRITICO").sum()
        n_low  = (alerts_df["alerta_stock"] == "BAJO").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Alertas Críticas", n_crit)
        c2.metric("🟡 Stock Bajo",       n_low)
        c3.metric("🟢 En OK",
                  (alerts_df["alerta_stock"] == "OK").sum())

        def color_alert(val):
            if val == "CRITICO":
                return "background-color:#f8d7da"
            if val == "BAJO":
                return "background-color:#fff3cd"
            return "background-color:#d4edda"

        st.dataframe(
            alerts_df.head(100)
            .style.map(color_alert, subset=["alerta_stock"]),
            use_container_width=True,
            height=280,
        )
    else:
        st.info("Ejecuta `python scripts/2_spark_processing.py` para generar las alertas.")

    st.markdown("---")

    top_df = load_processed("top_products_by_store.csv")
    if top_df is not None:
        st.subheader(f"Top 5 Productos — {sel_store}")
        store_top = top_df[top_df["tienda"] == sel_store]
        fig_top = px.bar(
            store_top, x="producto", y="total_ventas",
            color="categoria", title=f"Top 5 Productos — {sel_store}",
            labels={"total_ventas": "Ventas totales", "producto": "Producto"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_top.update_layout(height=340)
        st.plotly_chart(fig_top, use_container_width=True)


# TAB 4 — COMPARATIVA DE MODELOS
with tab4:
    st.subheader("Comparativa de Modelos de Predicción")

    if "results" in dir() and results:
        comp = compare_models(results)

        fig_cmp = make_subplots(
            rows=1, cols=3,
            subplot_titles=("MAPE (%)", "MAE (uds)", "RMSE (uds)"),
        )
        pal = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
        names = comp.index.tolist()

        for col_i, metric in enumerate(["MAPE (%)", "MAE", "RMSE"], start=1):
            vals = comp[metric].values
            fig_cmp.add_bar(
                x=names, y=vals,
                marker_color=pal[:len(names)],
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                showlegend=False,
                row=1, col=col_i,
            )
        fig_cmp.update_layout(height=380, title_text="Error por Modelo (Test set)")
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.dataframe(
            comp[["MAPE (%)", "MAE", "RMSE"]]
            .style.format("{:.2f}")
            .highlight_min(color="#d4edda")
            .highlight_max(color="#f8d7da"),
            use_container_width=True,
        )

        # Importancia de variables RF
        st.subheader("Importancia de Variables — Random Forest")
        fi = rf.get_feature_importance().head(12)
        fig_fi = px.bar(
            x=fi.values, y=fi.index, orientation="h",
            title="Top 12 Variables más Predictivas",
            labels={"x": "Importancia relativa", "y": "Variable"},
            color=fi.values, color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=380, yaxis=dict(autorange="reversed"),
                              showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.info(
            "**Interpretación:**  \n"
            "- **MAPE** (Mean Absolute Percentage Error): error porcentual medio. Objetivo < 15 %.  \n"
            "- **MAE**: error absoluto medio en unidades vendidas.  \n"
            "- **RMSE**: penaliza más los errores grandes.  \n"
            "- Las celdas en **verde** son los mejores valores; en **rojo**, los peores."
        )
    else:
        st.info("Selecciona tienda y producto en la pestaña 'Predicciones' para cargar los modelos.")


# Footer
st.markdown("---")
st.caption(
    "DataMind Retail · Proyecto Final del Curso de Especialización en IA y Big Data · 2025/26"
)
