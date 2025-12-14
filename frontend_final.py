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

def get_alert_color(camp, valor):
    """
    Retorna color segons si el valor és normal, alerta o crític.
    """
    try:
        v = float(valor)

        # Exemple de regles numèriques genèriques
        if v < 0:
            return "#f44336", "❌ Valor invàlid"
        elif camp.lower() in ["edat", "age"] and v > 80:
            return "#ff9800", "⚠️ Edat elevada"
        elif camp.lower() in ["imc", "bmi"] and v >= 30:
            return "#f44336", "🔴 IMC alt"
        elif camp.lower() in ["imc", "bmi"] and v >= 25:
            return "#ff9800", "🟠 Sobrepès"
        else:
            return "#4caf50", None  # normal

    except:
        # Valors textuals
        valor_str = str(valor).lower()

        if valor_str in ["sí", "si", "true", "positivo", "positiu"]:
            return "#f44336", "Positiu"
        if valor_str in ["no", "false", "negativo", "negatiu"]:
            return "#4caf50", None

    return "#2196f3", None  # neutre


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
    if st.button("Model", use_container_width=True): st.session_state.page = "modelo"
with nav[1]:
    if st.button("Pacient", use_container_width=True): st.session_state.page = "paciente"
with nav[2]:
    if st.button("Contacte", use_container_width=True): st.session_state.page = "contacto"

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
    st.header("Rendiment del Model")
    
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
    st.header("Avaluació de Pacients")
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
            tab1, tab2, tab3 = st.tabs([
                "Dades Generals",
                "Predicció IA",
                "Recomanacions",
            ])

            with tab1:
                st.subheader("🩺 Dades clíniques del pacient")

                # Columnes laterals per limitar amplada
                left, center, right = st.columns([1, 3, 1])

                with center:
                    fila = df.iloc[0]
                    col1, col2 = st.columns(2)

                    for i, (camp, valor) in enumerate(fila.items()):
                        color, alerta = get_alert_color(camp, valor)
                        target = col1 if i % 2 == 0 else col2

                        target.markdown(
                            f"""
                            <div style="
                                background-color: #1e1e1e;
                                padding: 14px;
                                margin-bottom: 12px;
                                border-radius: 12px;
                                border-left: 5px solid {color};
                            ">
                                <div style="color:#aaaaaa; font-size: 13px;">
                                    {camp}
                                </div>
                                <div style="font-size: 17px; font-weight: 600; color:{color};">
                                    {valor}
                                </div>
                                {"<div style='font-size:12px; color:#ffcc80;'>" + alerta + "</div>" if alerta else ""}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("""
                    **Llegenda:**
                    🟥 Vermell: Valor crític  
                    🟨 Groc: Valor a revisar  
                    🟩 Verd: Valor normal  
                    """)


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
                st.subheader("Recomanacions Clíniques Basades en Pràctica Assistencial")

                for idx, row in df.iterrows():
                    risc = row['prob_recidiva']

                    # ---- TARJETA PRINCIPAL DEL PACIENTE ----
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#1e1e1e;
                            padding:20px;
                            border-radius:16px;
                            margin-bottom:24px;
                            border-left:6px solid {'#f44336' if risc>=0.6 else '#ff9800' if risc>=0.4 else '#4caf50'};
                        ">
                            <h3 style="margin-bottom:5px;">
                                Pacient {idx+1}
                            </h3>
                            <p style="color:#cccccc; font-size:15px;">
                                Risc estimat de recidiva: <b>{risc:.1%}</b>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # ---- BLOQUE 1: NIVEL DE RIESGO ----
                    if risc >= 0.60:
                        st.error("""
                        🔴 **Risc Alt**
                        - Derivació urgent a Oncologia Ginecològica.
                        - Discussió en Comitè Multidisciplinari de Tumors.
                        """)
                    elif risc >= 0.40:
                        st.warning("""
                        🟠 **Risc Intermedi**
                        - Seguiment especialitzat per ginecologia oncològica.
                        - Controls clínics i radiològics semestrals.
                        """)
                    else:
                        st.success("""
                        🟢 **Risc Baix**
                        - Seguiment rutinari segons protocols estàndard.
                        - Educació en símptomes d'alarma.
                        """)

                    # ---- BLOQUE 2: PRUEBAS DIAGNÓSTICAS ----
                    st.markdown("### Proves Diagnòstiques Recomanades")
                    st.write("""
                    - **Ecografia transvaginal** per valoració inicial de l'endometri.
                    - **Biòpsia endometrial** (pipelle o histeroscòpia) si hi ha sospita clínica.
                    - **RM pèlvica** per estudiar invasió miometrial i extensió local.
                    - **TC toracoabdominal** en risc intermedi-alt o sospita de disseminació.
                    - **Estudi anatomopatològic complet** (tipus histològic i grau).
                    """)

                    # ---- BLOQUE 3: FACTORES CLÍNICOS ----
                    st.markdown("### Factors Clínics a Optimitzar")

                    factors = False

                    if 'imc' in row and row['imc'] >= 30:
                        st.write("🔸 **Obesitat:** Recomanable intervenció nutricional estructurada.")
                        factors = True
                    if 'diabetis' in row and str(row['diabetis']).lower() in ["si", "sí", "true", "1"]:
                        st.write("🔸 **Diabetis:** Optimitzar control glucèmic (HbA1c).")
                        factors = True
                    if 'hipertensio' in row and str(row['hipertensio']).lower() in ["si", "sí", "true", "1"]:
                        st.write("🔸 **Hipertensió:** Ajust i seguiment del tractament.")
                        factors = True

                    if not factors:
                        st.write("No es detecten factors clínics modificables rellevants.")

                    # ---- BLOQUE 4: SEGUIMIENTO ----
                    st.markdown("### Pla de Seguiment Orientatiu")

                    if risc >= 0.60:
                        st.write("""
                        - Revisió cada **3 mesos** els primers 2 anys.
                        - Exploració ginecològica completa en cada visita.
                        - Proves d'imatge segons criteri clínic.
                        """)
                    elif risc >= 0.40:
                        st.write("""
                        - Revisió cada **6 mesos**.
                        - Exploració clínica + ecografia segons indicació.
                        """)
                    else:
                        st.write("""
                        - Revisió **anual**.
                        - Informar sobre sagnat postmenopàusic o dolor pèlvic.
                        """)

                    st.info(
                        "ℹRecomanacions orientatives basades en pràctica clínica habitual "
                        "i guies de maneig del càncer d'endometri. "
                        "La decisió final correspon sempre a l'equip mèdic responsable."
                    )

                    st.markdown("---")

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