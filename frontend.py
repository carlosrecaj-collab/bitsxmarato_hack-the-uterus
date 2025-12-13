import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import shap

# =====================================================
# 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO)
# =====================================================
st.set_page_config(
    page_title="EndoPredict Pro",
    page_icon="⚕️",
    layout="wide"
)

# =====================================================
# 2. CARGAR MODELO Y ARTEFACTOS
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        # Cargamos el diccionario completo (modelo, scaler, kmeans, etc.)
        artifacts = joblib.load('xgb_clinic_model_robust.joblib')
        return artifacts
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'xgb_clinic_model_robust.joblib'. Asegúrate de que está en la misma carpeta.")
        return None

artifacts = load_artifacts()

# =====================================================
# 3. FUNCIÓN DE PREDICCIÓN (CON SHAP)
# =====================================================
def fer_prediccio_completa(row_data, artifacts):
    """
    Procesa una fila, predice probabilidad y calcula el factor principal con SHAP.
    """
    # a. Preparar DataFrame de una fila
    df_single = pd.DataFrame([row_data]).fillna(-1)
    
    # b. Encoding
    object_cols = df_single.select_dtypes(include=['object']).columns
    df_single = pd.get_dummies(df_single, columns=object_cols, drop_first=True, dtype=int)
    
    # c. KMeans Feature Engineering
    kmeans = artifacts.get('kmeans')
    scaler = artifacts.get('scaler')
    vars_kmeans = artifacts.get('vars_kmeans', [])
    
    if kmeans and scaler and vars_kmeans:
        X_km = df_single.copy()
        for col in vars_kmeans:
            if col not in X_km.columns:
                X_km[col] = -1 
        
        # Filtrar solo columnas necesarias para kmeans
        X_km = X_km[vars_kmeans]
        X_km_s = scaler.transform(X_km)
        
        df_single['cluster_group'] = kmeans.predict(X_km_s)
        dists = kmeans.transform(X_km_s)
        for i in range(dists.shape[1]):
            df_single[f'dist_cluster_{i}'] = dists[:, i]

    # d. Alineación de columnas
    expected_cols = artifacts['features']
    df_final = df_single.reindex(columns=expected_cols, fill_value=0)
    
    # e. Predicción
    model = artifacts['model'] # Este es el CalibratedClassifierCV
    threshold = artifacts.get('threshold', 0.5)
    
    prob = model.predict_proba(df_final)[:, 1][0]
    pred_class = 1 if prob >= threshold else 0
    
    # f. SHAP (Factor Principal)
    # Necesitamos el estimador base (XGBoost) dentro del modelo calibrado
    base_model = model.base_estimator if hasattr(model, 'base_estimator') else model
    
    factor_principal = "Análisis complejo"
    try:
        explainer = shap.TreeExplainer(base_model)
        # SHAP values para esta fila
        shap_values = explainer.shap_values(df_final)
        
        # Manejo de formato de salida de SHAP (lista o array)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Encontrar el índice del valor absoluto más alto
        top_idx = np.argmax(np.abs(vals))
        col_name = expected_cols[top_idx]
        impacto = vals[top_idx]
        
        signo = "(+)" if impacto > 0 else "(-)" # + Aumenta riesgo, - Disminuye
        factor_principal = f"{col_name} {signo}"
        
    except Exception as e:
        factor_principal = "No disponible"

    return prob, pred_class, factor_principal

# =====================================================
# 4. INTERFAZ: NAVEGACIÓN
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "modelo"

nav = st.columns(4)

def nav_button(label, page_name):
    active = "nav-btn-active" if st.session_state.page == page_name else ""
    if nav[list(nav).index(next(filter(lambda c: c, nav)))].button(label, key=f"nav_{page_name}"): # Hack simple para botones
         st.session_state.page = page_name

with nav[0]:
    if st.button("📊 Modelo", use_container_width=True): st.session_state.page = "modelo"
with nav[1]:
    if st.button("🔬 Paciente", use_container_width=True): st.session_state.page = "paciente"
with nav[2]:
    if st.button("📚 Metodología", use_container_width=True): st.session_state.page = "metodologia"
with nav[3]:
    if st.button("📧 Contacto", use_container_width=True): st.session_state.page = "contacto"

st.markdown("---")

# HEADER
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.title("EndoPredict Pro: Evaluación de Riesgo")
    st.markdown("### *Claridad predictiva para decisiones clínicas informadas*")

# =====================================================
# 5. CONTENIDO DE PÁGINAS
# =====================================================
page = st.session_state.page

# --- MODELO ---
if page == "modelo":
    st.header("📊 Rendimiento del Modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC ROC Test", "0.92")
    c2.metric("Sensibilidad", "91%")
    c3.metric("Brier Score", "0.08")
    st.info("Modelo XGBoost entrenado con validación cruzada estratificada repetida (5x10).")

# --- PACIENTE (LÓGICA PRINCIPAL) ---
elif page == "paciente":
    st.header("🔬 Evaluación de Pacientes")
    st.write("Sube un archivo CSV con los datos clínicos.")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

    if uploaded_file is not None and artifacts is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # ---------------------------------------------------------
            # PASO CRÍTICO: PROCESAMIENTO EN LOTE (BATCH)
            # Calculamos las predicciones ANTES de mostrar las pestañas
            # ---------------------------------------------------------
            probs = []
            classes = []
            factors = []
            
            with st.spinner('Analizando pacientes con IA...'):
                for idx, row in df.iterrows():
                    p, c, f = fer_prediccio_completa(row, artifacts)
                    probs.append(p)
                    classes.append(c)
                    factors.append(f)
            
            # Guardamos resultados en el DF para usarlos en todas las pestañas
            df['prob_recidiva'] = probs
            df['pred_clase'] = classes # 1 = Recidiva, 0 = No
            df['factor_principal'] = factors
            
            st.success(f"Análisis completado para {len(df)} pacientes.")

            # ---------------- PESTAÑAS ----------------
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Datos Generales",
                "🤖 Predicción IA",
                "🩺 Recomendaciones",
                "📈 Gráficos"
            ])

            # TAB 1: DATOS
            with tab1:
                st.subheader("Datos clínicos cargados")
                st.dataframe(df)

            # TAB 2: PREDICCIÓN DETALLADA
            with tab2:
                st.subheader("Análisis de Riesgo Individual")
                threshold = artifacts.get('threshold', 0.5)
                
                for idx, row in df.iterrows():
                    st.markdown(f"#### Paciente {idx+1}")
                    
                    c1, c2, c3 = st.columns(3)
                    
                    # Métricas
                    c1.metric("Probabilidad Recidiva", f"{row['prob_recidiva']:.1%}")
                    
                    estado = "ALTO RIESGO" if row['pred_clase'] == 1 else "BAJO RIESGO"
                    icono = "⚠️" if row['pred_clase'] == 1 else "✅"
                    c2.metric("Clasificación", f"{icono} {estado}")
                    
                    c3.metric("Factor Principal", row['factor_principal'])
                    
                    # Barra visual
                    st.progress(int(row['prob_recidiva'] * 100))
                    st.divider()

            # TAB 3: RECOMENDACIONES
            with tab3:
                st.subheader("Recomendaciones Clínicas")
                for idx, row in df.iterrows():
                    with st.expander(f"Paciente {idx+1} ({row['prob_recidiva']:.1%})"):
                        rec = []
                        
                        # Riesgo
                        if row['prob_recidiva'] > 0.60:
                            rec.append("🔴 **Prioridad Alta:** Derivación urgente a oncología y pruebas de imagen.")
                        elif row['prob_recidiva'] > threshold:
                            rec.append("🟠 **Riesgo Moderado:** Seguimiento estrecho trimestral.")
                        else:
                            rec.append("🟢 **Riesgo Bajo:** Controles rutinarios.")
                        
                        # IMC (si existe la columna)
                        if 'imc' in row:
                            if row['imc'] > 30:
                                rec.append("📉 **Nutrición:** Se recomienda plan de pérdida de peso (IMC > 30).")
                        
                        # Factor principal
                        rec.append(f"ℹ️ **Atención a:** {row['factor_principal']} (Factor determinante en el modelo).")
                        
                        for r in rec:
                            st.write(r)

            # TAB 4: GRÁFICOS GLOBALES
            with tab4:
                st.subheader("Panorama del Grupo de Pacientes")
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("**Distribución de Riesgo (Probabilidad)**")
                    st.bar_chart(df['prob_recidiva'])
                
                with c2:
                    st.write("**Clasificación (Recidiva vs No)**")
                    counts = df['pred_clase'].map({0: 'Bajo Riesgo', 1: 'Alto Riesgo'}).value_counts()
                    st.bar_chart(counts)

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
            st.write("Por favor revisa que el CSV tenga el formato correcto.")

# --- OTRAS PÁGINAS ---
elif page == "metodologia":
    st.header("📚 Metodología")
    st.write("El modelo utiliza XGBoost con imputación KMeans y validación cruzada repetida.")

elif page == "contacto":
    st.header("📧 Contacto")
    st.write("Soporte técnico: soporte@endopredict.com")

st.markdown("---")
st.markdown("© 2025 EndoPredict Pro")