import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import shap

# --- CONFIGURACIÓ DE LA PÀGINA ---
st.set_page_config(
    page_title="EndoPredict Pro",
    page_icon="⚕️",
    layout="wide"
)

# --- GESTIÓ DE RECURSOS (CACHING) ---
@st.cache_resource
def load_artifacts():
    """
    Carrega el fitxer .joblib que conté el model entrenat i els objectes de preprocessament.
    
    Returns:
        dict: Diccionari amb 'model', 'scaler', 'kmeans', 'features', etc.
        None: Si no es troba el fitxer.
    """
    try:
        # Carreguem el diccionari complet generat en l'entrenament
        artifacts = joblib.load('xgb_clinic_model_robust.joblib')
        return artifacts
    except FileNotFoundError:
        st.error("⚠️ Error crític: No s'ha trobat el fitxer 'xgb_clinic_model_robust.joblib'. Verifica el directori.")
        return None

def process_patient_prediction(row_data, artifacts):
    """
    Processa les dades d'un sol pacient, aplica enginyeria de característiques (KMeans),
    realitza la predicció i calcula l'explicabilitat (SHAP).

    Args:
        row_data (pd.Series): Fila amb les dades del pacient.
        artifacts (dict): Diccionari amb els objectes del model.

    Returns:
        tuple: (probabilitat, classe_predita, factor_principal_text)
    """
    # 1. Preprocessament bàsic
    df_single = pd.DataFrame([row_data]).fillna(-1)
    
    # One-Hot Encoding (simulat per adaptar-se a l'estructura d'entrada)
    object_cols = df_single.select_dtypes(include=['object']).columns
    df_single = pd.get_dummies(df_single, columns=object_cols, drop_first=True, dtype=int)
    
    # 2. Feature Engineering: KMeans (si aplica)
    kmeans = artifacts.get('kmeans')
    scaler = artifacts.get('scaler')
    vars_kmeans = artifacts.get('vars_kmeans', [])
    
    if kmeans and scaler and vars_kmeans:
        # Creem còpia per al càlcul de clústers
        X_km = df_single.copy()
        
        # Assegurem que existeixen les columnes necessàries, si no, imputem -1
        for col in vars_kmeans:
            if col not in X_km.columns:
                X_km[col] = -1 
        
        X_km = X_km[vars_kmeans]
        X_km_s = scaler.transform(X_km)
        
        # Assignació de clúster i distàncies
        df_single['cluster_group'] = kmeans.predict(X_km_s)
        dists = kmeans.transform(X_km_s)
        for i in range(dists.shape[1]):
            df_single[f'dist_cluster_{i}'] = dists[:, i]

    # 3. Alineació de columnes amb el model entrenat
    expected_cols = artifacts['features']
    df_final = df_single.reindex(columns=expected_cols, fill_value=0)
    
    # 4. Predicció
    model = artifacts['model'] 
    threshold = artifacts.get('threshold', 0.5)
    
    prob = model.predict_proba(df_final)[:, 1][0]
    pred_class = 1 if prob >= threshold else 0
    
    # 5. Explicabilitat (SHAP)
    # Accedim a l'estimador base (XGBoost) ja que el model calibrat no té TreeExplainer directe
    base_model = model.base_estimator if hasattr(model, 'base_estimator') else model
    
    factor_principal = "Anàlisi complexa"
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(df_final)
        
        # Gestió de dimensions de sortida de SHAP
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Identificació de la variable amb major impacte absolut
        top_idx = np.argmax(np.abs(vals))
        col_name = expected_cols[top_idx]
        impacto = vals[top_idx]
        
        signo = "(+)" if impacto > 0 else "(-)" # (+) Augmenta risc, (-) Disminueix risc
        factor_principal = f"{col_name} {signo}"
        
    except Exception:
        factor_principal = "No disponible"

    return prob, pred_class, factor_principal

# --- CÀRREGA INICIAL ---
artifacts = load_artifacts()

# --- NAVEGACIÓ I ESTRUCTURA ---
if "page" not in st.session_state:
    st.session_state.page = "modelo"

# Menú de navegació superior
nav = st.columns(3)
with nav[0]:
    if st.button("📊 Model", use_container_width=True): st.session_state.page = "modelo"
with nav[1]:
    if st.button("🔬 Pacient", use_container_width=True): st.session_state.page = "paciente"
with nav[2]:
    if st.button("📧 Contacte", use_container_width=True): st.session_state.page = "contacto"

st.markdown("---")

# Capçalera principal
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.title("EndoPredict Pro: Avaluació de Risc")
    st.markdown("### *Claredat predictiva per a decisions clíniques informades*")

page = st.session_state.page

# =====================================================
# PÀGINA 1: DASHBOARD DEL MODEL
# =====================================================
if page == "modelo":
    st.header("📊 Rendiment del Model")
    
    # Recuperació de mètriques emmagatzemades o valors per defecte
    metrics_saved = artifacts.get('metrics', {}) if artifacts else {}
    
    # Valors de referència (Fallback)
    auc_val = metrics_saved.get('auc_test', 0.887)
    sens_val = metrics_saved.get('sensitivity', 0.83)
    brier_val = metrics_saved.get('brier_score', 0.116)

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC ROC Test", f"{auc_val:.2f}")
    c2.metric("Sensibilitat", f"{sens_val:.0%}")
    c3.metric("Brier Score", f"{brier_val:.2f}")
    
    st.info("Model XGBoost entrenat amb validació creuada estratificada repetida (5 folds x 10 repeticions). Calibratge isotònic aplicat.")

# =====================================================
# PÀGINA 2: AVALUACIÓ DE PACIENTS
# =====================================================
elif page == "paciente":
    st.header("🔬 Avaluació de Pacients")
    st.write("Carrega un fitxer CSV amb les dades clíniques per analitzar.")

    uploaded_file = st.file_uploader("Puja un fitxer CSV", type="csv")

    if uploaded_file is not None and artifacts is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Emmagatzematge de resultats
            probs = []
            classes = []
            factors = []
            
            with st.spinner('Analitzant pacients amb IA...'):
                for idx, row in df.iterrows():
                    p, c, f = process_patient_prediction(row, artifacts)
                    probs.append(p)
                    classes.append(c)
                    factors.append(f)
            
            # Assignació de resultats al DataFrame
            df['prob_recidiva'] = probs
            df['pred_clase'] = classes
            df['factor_principal'] = factors
            
            st.success(f"Anàlisi completada per a {len(df)} pacients.")

            # Pestanyes de visualització
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Dades Generals",
                "🤖 Predicció IA",
                "🩺 Recomanacions",
                "📈 Gràfics"
            ])

            with tab1:
                st.subheader("Dades clíniques carregades")
                st.dataframe(df)

            with tab2:
                st.subheader("Anàlisi de Risc Individual")
                
                for idx, row in df.iterrows():
                    st.markdown(f"#### Pacient {idx+1}")
                    
                    c1, c2, c3 = st.columns(3)
                    
                    c1.metric("Probabilitat Recidiva", f"{row['prob_recidiva']:.1%}")
                    
                    estado = "ALT RISC" if row['pred_clase'] == 1 else "BAIX RISC"
                    icono = "⚠️" if row['pred_clase'] == 1 else "✅"
                    c2.metric("Classificació", f"{icono} {estado}")
                    
                    c3.metric("Factor Principal", row['factor_principal'])
                    
                    st.progress(int(row['prob_recidiva'] * 100))
                    st.divider()

            with tab3:
                st.subheader("Recomanacions Clíniques")
                threshold = artifacts.get('threshold', 0.5)

                for idx, row in df.iterrows():
                    with st.expander(f"Pacient {idx+1} ({row['prob_recidiva']:.1%})"):
                        rec = []
                        
                        # Lògica de regles de negoci basada en el risc
                        if row['prob_recidiva'] > 0.60:
                            rec.append("🔴 **Prioritat Alta:** Derivació urgent a oncologia i proves d'imatge.")
                        elif row['prob_recidiva'] > threshold:
                            rec.append("🟠 **Risc Moderat:** Seguiment estret trimestral.")
                        else:
                            rec.append("🟢 **Risc Baix:** Controls rutinaris.")
                        
                        # Exemple de regla addicional (IMC)
                        if 'imc' in row and row['imc'] > 30:
                            rec.append("📉 **Nutrició:** Es recomana pla de pèrdua de pes (IMC > 30).")
                        
                        rec.append(f"ℹ️ **Atenció a:** {row['factor_principal']} (Factor determinant segons SHAP).")
                        
                        for r in rec:
                            st.write(r)

            with tab4:
                st.subheader("Visió Global de la Cohort") 
                
                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("**Distribució de Risc (Probabilitat)**")
                    st.bar_chart(df['prob_recidiva'])
                
                with c2:
                    st.write("**Classificació (Recidiva vs No)**")
                    counts = df['pred_clase'].map({0: 'Baix Risc', 1: 'Alt Risc'}).value_counts()
                    st.bar_chart(counts)

        except Exception as e:
            st.error(f"Error processant el fitxer: {e}")
            st.write("Verifica que el CSV tingui el format correcte i les columnes esperades.")

# =====================================================
# PÀGINA 3: CONTACTE
# =====================================================
elif page == "contacto":
    st.header("📧 Contacte")
    st.write("Suport tècnic: suport@endopredict.com")

# Peu de pàgina
st.markdown("---")
st.markdown("© 2025 EndoPredict Pro")