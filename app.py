"""
══════════════════════════════════════════════════════════╗
║         MELATE FORECASTER - App Web (iPhone/PC)         ║
╚══════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import sys

# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Melate Forecaster",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CSV_HISTORICO, RANGO_NUMEROS, NUM_BOLAS_POR_SORTEO,
    N_COMBINACIONES_RECOMENDADAS, COSTO_BOLETO, OUTPUT_DIR
)

# ═══════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.3rem !important; }
        .stButton>button { width: 100%; }
    }
    .rec-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 15px; color: white;
        margin: 0.8rem 0; box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }
    .rec-card h3 { margin: 0 0 0.5rem 0; color: white; }
    .ball {
        display: inline-block; width: 42px; height: 42px;
        line-height: 42px; text-align: center; background: white;
        color: #667eea; border-radius: 50%; font-weight: bold;
        margin: 0 4px; font-size: 1.1rem;
    }
    @media (max-width: 768px) {
        .ball { width: 36px; height: 36px; line-height: 36px; font-size: 1rem; }
    }
    .alert-info {
        background: #1e3a5f; border-left: 4px solid #2196f3;
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
    }
    .alert-warning {
        background: #5f4e1e; border-left: 4px solid #ff9800;
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  CARGA DE DATOS ROBUSTA
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_cache():
    if not CSV_HISTORICO.exists():
        return None

    df = pd.read_csv(CSV_HISTORICO)

    # 1. Asegurar que exista la columna 'fecha'
    if "fecha" not in df.columns:
        for col in df.columns:
            if "fecha" in col.lower() or "date" in col.lower():
                df = df.rename(columns={col: "fecha"})
                break
    
    if "fecha" not in df.columns:
        return None

    # 2. Limpiar espacios en la columna fecha
    df["fecha"] = df["fecha"].astype(str).str.strip()

    # 3. RECONSTRUIR fecha_dt si está vacía (por la coma extra del CSV)
    if "fecha_dt" in df.columns:
        # Intentamos leer fecha_dt. Si está vacía, se convierte en NaT
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
        
        # Donde sea NaT (vacío), usamos la columna 'fecha' para rellenarlo
        filas_vacias = df["fecha_dt"].isna()
        if filas_vacias.any():
            df.loc[filas_vacias, "fecha_dt"] = pd.to_datetime(
                df.loc[filas_vacias, "fecha"], dayfirst=True, errors="coerce"
            )
    else:
        # Si la columna no existe, la creamos desde 'fecha'
        df["fecha_dt"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")

    # 4. Eliminar solo las filas que realmente no tengan fecha válida y ordenar
    df = df.dropna(subset=["fecha_dt"])
    df = df.drop_duplicates(subset=["fecha_dt"]).sort_values("fecha_dt").reset_index(drop=True)

    return df if not df.empty else None



def bolas_html(numeros):
    return " ".join(f'<span class="ball">{n:02d}</span>' for n in sorted(numeros))


def mostrar_recomendacion(rank, combinacion, score):
    nums = bolas_html(combinacion)
    st.markdown(f"""
    <div class="rec-card">
        <h3>Combinación #{rank}</h3>
        <div style="margin: 0.8rem 0;">{nums}</div>
        <div style="opacity: 0.9; font-size: 0.9rem;">Score: {score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.title(" Melate Forecaster")
st.sidebar.markdown("---")

df = cargar_datos_cache()
n_sorteos = len(df) if df is not None else 0

st.sidebar.metric("Sorteos cargados", n_sorteos)
if n_sorteos > 0:
    try:
        ultima_fecha = df['fecha_dt'].max()
        st.sidebar.caption(f"Último: {ultima_fecha.strftime('%d/%m/%Y')}")
    except:
        st.sidebar.caption("Fecha no disponible")

st.sidebar.markdown("---")
opcion = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🚀 Correr Todo", "📊 Dashboard", "🔍 Análisis", 
     "🎯 Recomendaciones", "📈 Backtest", "📱 Instalar en iPhone"]
)


# ═══════════════════════════════════════════════════════════
#  PÁGINA: INICIO
# ═══════════════════════════════════════════════════════════
if opcion == "🏠 Inicio":
    st.title(" Melate Forecaster")
    st.markdown("Sistema de análisis y recomendaciones para el Melate.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sorteos", n_sorteos)
    with col2:
        st.metric("Números", f"1-{RANGO_NUMEROS}")
    with col3:
        st.metric("Costo boleto", f"${COSTO_BOLETO} MXN")

    if df is None or df.empty:
        st.warning("⚠️ No hay datos cargados. Ve a **🚀 Correr Todo** para descargar el histórico.")
    else:
        st.markdown("---")
        st.markdown("""
        <div class="alert-info">
            <b>💡 Cómo usar:</b><br>
            1. Ve a <b>🚀 Correr Todo</b> para actualizar datos y generar predicción.<br>
            2. Revisa <b>🔍 Análisis</b> para ver patrones estadísticos.<br>
            3. <b>📱 Instalar en iPhone</b> para usar como app nativa.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-warning">
            <b>️ Aviso importante:</b> El Melate es un sorteo aleatorio.
            Ningún modelo puede predecir los números con certeza. Este sistema
            optimiza combinaciones balanceadas y evita números populares para
            <b>maximizar el premio esperado</b> (no la probabilidad de ganar).
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PÁGINA: CORRER TODO
# ═══════════════════════════════════════════════════════════
elif opcion == "🚀 Correr Todo":
    st.title("🚀 Correr Todo el Proceso")
    st.markdown("Este botón ejecuta: **Descargar datos → Generar predicción**")

    st.markdown("""
    <div class="alert-info">
        <b>¿Qué hace este botón?</b><br>
        1. 📥 Descarga los últimos resultados del Melate<br>
        2. 🧠 Usa el modelo entrenado para generar predicciones<br>
        3. 🎯 Te muestra 5 combinaciones optimizadas
    </div>
    """, unsafe_allow_html=True)

    if st.button(" EJECUTAR TODO", type="primary", use_container_width=True):
        
        log_container = st.container()
        
        with log_container:
            st.subheader("📋 Logs de ejecución:")
            
            # PASO 1: Descargar datos
            st.write("**Paso 1/2:** Descargando datos...")
            
            try:
                from scraper import descargar_csv_oficial, estandarizar_dataframe, cargar_csv_existente, guardar_csv
                
                st.write("  → Importando funciones del scraper... ✅")
                
                df_crudo = descargar_csv_oficial()
                st.write(f"  → CSV descargado: {len(df_crudo)} filas")
                
                if not df_crudo.empty:
                    df_nuevo = estandarizar_dataframe(df_crudo)
                    st.write(f"  → Datos estandarizados: {len(df_nuevo)} sorteos")
                    
                    if not df_nuevo.empty:
                        df_existente = cargar_csv_existente()
                        st.write(f"  → Datos existentes: {len(df_existente)} sorteos")
                        
                        if not df_existente.empty:
                            df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
                            df_final = df_final.drop_duplicates(subset=["fecha"]).sort_values("fecha_dt").reset_index(drop=True)
                        else:
                            df_final = df_nuevo
                        
                        guardar_csv(df_final)
                        st.success(f"✅ {len(df_nuevo)} sorteos nuevos guardados")
                    else:
                        st.warning("⚠️ CSV descargado pero no se pudo procesar")
                else:
                    st.warning("⚠️ No se pudieron descargar datos nuevos, usando histórico")
                
                st.cache_data.clear()
                df = cargar_datos_cache()
                st.write(f"  → Datos cargados en caché: {len(df) if df is not None else 0} sorteos")
                
            except Exception as e:
                st.error(f"❌ Error en Paso 1: {e}")
                st.exception(e)
                st.stop()
            
            st.write("\n**Paso 2/2:** Generando predicción...")
            
            try:
                from generador import generar_recomendaciones
                
                st.write("  → Importando generador... ✅")
                
                combinaciones = generar_recomendaciones(N_COMBINACIONES_RECOMENDADAS)
                st.session_state["recomendaciones"] = combinaciones
                
                st.success(f"✅ {len(combinaciones)} combinaciones generadas")
                
            except Exception as e:
                st.error(f"❌ Error en Paso 2: {e}")
                st.exception(e)
                st.stop()
        
        st.markdown("---")
        st.subheader("🎯 Tus combinaciones para el próximo sorteo:")
        
        if "recomendaciones" in st.session_state:
            for i, (comb, score) in enumerate(st.session_state["recomendaciones"], 1):
                mostrar_recomendacion(i, comb, score)
            
            csv_data = []
            for i, (comb, score) in enumerate(st.session_state["recomendaciones"], 1):
                row = {"Rank": i, "Score": f"{score:.4f}"}
                for j, n in enumerate(comb, 1):
                    row[f"N{j}"] = n
                csv_data.append(row)
            csv_bytes = pd.DataFrame(csv_data).to_csv(index=False).encode("utf-8")
            
            st.download_button(
                "📥 Descargar CSV",
                csv_bytes,
                f"melate_recomendaciones_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.warning("️ No se generaron combinaciones")


# ═══════════════════════════════════════════════════════════
#  PÁGINA: DASHBOARD
# ═══════════════════════════════════════════════════════════
elif opcion == "📊 Dashboard":
    st.title("📊 Dashboard General")

    if df is None or df.empty:
        st.warning("️ No hay datos. Ve a **🚀 Correr Todo** para cargar datos.")
    else:
        try:
            cols_num = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]
            df["suma"] = df[cols_num].sum(axis=1)

            nums = df[cols_num].values.flatten()
            freq, _ = np.histogram(nums, bins=range(1, RANGO_NUMEROS + 2))
            df_freq = pd.DataFrame({
                "Número": range(1, RANGO_NUMEROS + 1),
                "Frecuencia": freq
            })

            fig1 = px.bar(df_freq, x="Número", y="Frecuencia",
                          title="Frecuencia de cada número",
                          color="Frecuencia", color_continuous_scale="Viridis")
            fig1.add_hline(y=freq.mean(), line_dash="dash", line_color="red",
                           annotation_text="Esperado")
            st.plotly_chart(fig1, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.histogram(df, x="suma", nbins=30,
                                    title="Distribución de la suma")
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                top10 = df_freq.nlargest(10, "Frecuencia")
                fig3 = px.bar(top10, x="Número", y="Frecuencia",
                              title="Top 10 más frecuentes",
                              color="Frecuencia", color_continuous_scale="Plasma")
                st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.line(df, x="fecha_dt", y="suma",
                           title="Suma por sorteo (serie temporal)")
            fig4.add_hline(y=df["suma"].mean(), line_dash="dash", line_color="red")
            st.plotly_chart(fig4, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error al generar gráficas: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════
#  PÁGINA: ANÁLISIS
# ═══════════════════════════════════════════════════════════
elif opcion == "🔍 Análisis":
    st.title("🔍 Análisis Estadístico")

    if df is None or df.empty:
        st.warning("⚠️ No hay datos cargados.")
    else:
        try:
            from analisis import test_uniformidad_chi2, test_paridad

            chi2, p_chi = test_uniformidad_chi2(df)
            _, p_par = test_paridad(df)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("χ² Uniformidad", f"{chi2:.2f}")
                st.caption(f"p-valor: {p_chi:.4f}")
                if p_chi < 0.05:
                    st.error("⚠️ Posible sesgo detectado")
                else:
                    st.success("✓ Máquina parece justa")

            with col2:
                st.metric("χ² Paridad", f"{p_par:.4f}")
                st.caption("p-valor del test par/impar")
                if p_par < 0.05:
                    st.warning("️ Desequilibrio par/impar")
                else:
                    st.success("✓ Paridad balanceada")

            st.markdown("---")
            st.subheader("Números calientes y fríos")

            cols_num = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]
            nums = df[cols_num].values.flatten()
            freq, _ = np.histogram(nums, bins=range(1, RANGO_NUMEROS + 2))
            df_freq = pd.DataFrame({
                "Número": range(1, RANGO_NUMEROS + 1),
                "Frecuencia": freq
            }).sort_values("Frecuencia", ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔥 Top 10 calientes**")
                st.dataframe(df_freq.head(10), use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**❄️ Top 10 fríos**")
                st.dataframe(df_freq.tail(10).sort_values("Frecuencia"),
                            use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"❌ Error en análisis: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════
#  PÁGINA: RECOMENDACIONES
# ═══════════════════════════════════════════════════════════
elif opcion == "🎯 Recomendaciones":
    st.title("🎯 Recomendaciones")

    if df is None or df.empty:
        st.warning("⚠️ No hay datos cargados. Ve a 🚀 Correr Todo primero.")
    else:
        n_comb = st.slider("Número de combinaciones", 1, 10,
                           N_COMBINACIONES_RECOMENDADAS)

        if st.button("🎲 Generar recomendaciones", type="primary",
                     use_container_width=True):
            with st.spinner("Generando combinaciones optimizadas..."):
                try:
                    from generador import generar_recomendaciones
                    combinaciones = generar_recomendaciones(n_comb)
                    st.session_state["recomendaciones"] = combinaciones
                    st.success(f"✅ {len(combinaciones)} combinaciones generadas")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

        if "recomendaciones" in st.session_state:
            for i, (comb, score) in enumerate(st.session_state["recomendaciones"], 1):
                mostrar_recomendacion(i, comb, score)

            csv_data = []
            for i, (comb, score) in enumerate(st.session_state["recomendaciones"], 1):
                row = {"Rank": i, "Score": f"{score:.4f}"}
                for j, n in enumerate(comb, 1):
                    row[f"N{j}"] = n
                csv_data.append(row)
            csv_bytes = pd.DataFrame(csv_data).to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Descargar CSV",
                csv_bytes,
                f"melate_recomendaciones_{datetime.now():%Y%m%d_%H%M}.csv",
                "text/csv",
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════════
#  PÁGINA: BACKTEST
# ═══════════════════════════════════════════════════════════
elif opcion == "📈 Backtest":
    st.title("📈 Backtest de Estrategias")

    if df is None or len(df) < 50:
        st.warning("⚠️ Se necesitan al menos 50 sorteos para backtest.")
    else:
        n_sorteos_bt = st.slider("Sorteos a simular", 50, min(len(df), 500),
                                 min(200, len(df)))
        n_sims = st.slider("Simulaciones", 3, 20, 5)

        if st.button("🚀 Ejecutar backtest", type="primary",
                     use_container_width=True):
            try:
                from backtest import cargar_modelo, simular_estrategia
                modelos, feature_cols = cargar_modelo()

                progreso = st.progress(0)
                resultados = {}
                estrategias = ["aleatorio", "frecuencia"]
                if modelos is not None:
                    estrategias.insert(0, "modelo")

                for i, estr in enumerate(estrategias):
                    rois = []
                    for sim in range(n_sims):
                        r = simular_estrategia(df, estr, n_sorteos_bt,
                                               modelos, feature_cols, semilla=sim)
                        rois.append(r["roi"])
                    resultados[estr] = {
                        "roi_promedio": np.mean(rois),
                        "roi_std": np.std(rois),
                        "rois": rois
                    }
                    progreso.progress((i + 1) / len(estrategias))

                st.subheader("Resultados comparativos")
                df_resultados = pd.DataFrame([
                    {"Estrategia": k,
                     "ROI promedio (%)": f"{v['roi_promedio']:+.2f}",
                     "ROI std (%)": f"{v['roi_std']:.2f}"}
                    for k, v in resultados.items()
                ])
                st.dataframe(df_resultados, use_container_width=True, hide_index=True)

                fig = go.Figure()
                for estr, data in resultados.items():
                    fig.add_trace(go.Box(y=data["rois"], name=estr))
                fig.update_layout(
                    title="Distribución de ROI por estrategia",
                    yaxis_title="ROI (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f" Error en backtest: {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════
#  PÁGINA: INSTALAR EN iPHONE
# ═══════════════════════════════════════════════════════════
elif opcion == "📱 Instalar en iPhone":
    st.title("📱 Instalar en tu iPhone")

    st.markdown("""
    <div class="alert-info">
        Puedes usar esta app en tu iPhone como app nativa, sin App Store.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🚀 Paso 1: Desplegar en Streamlit Cloud (gratis)")
    st.markdown("""
    1. Sube tu proyecto a **GitHub** (incluyendo el archivo `modelo_xgb.joblib`)
    2. Ve a [share.streamlit.io](https://share.streamlit.io)
    3. Conecta tu cuenta de GitHub
    4. Selecciona tu repositorio y el archivo `app.py`
    5. Click en **Deploy**
    6. En 2 minutos tendrás una URL pública
    """)

    st.subheader("📲 Paso 2: Instalar en iPhone")
    st.markdown("""
    1. Abre **Safari** en tu iPhone (️ solo Safari, no Chrome)
    2. Ve a la URL de tu app desplegada
    3. Toca el botón **Compartir** ⬆️ (abajo en el centro)
    4. Desliza hacia abajo y toca **"Agregar a pantalla de inicio"**
    5. Edita el nombre si quieres (ej: "Melate")
    6. Toca **"Agregar"**

    ✅ ¡Listo! Tendrás un ícono en tu pantalla de inicio que abre la app
    en pantalla completa, como una app nativa.
    """)

    st.subheader("💡 ¿Cómo funciona?")
    st.markdown("""
    - **En tu PC**: Usas `python main.py --todo` para entrenar el modelo (una vez al mes)
    - **En tu iPhone**: Usas el botón " Correr Todo" para actualizar datos y generar predicciones (todos los días)
    - **La web**: Lee el modelo que ya entrenaste y solo hace predicciones ligeras
    """)