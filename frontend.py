import streamlit as st
import pandas as pd
import numpy as np
import requests

# =====================================================
# CONFIGURACIÓN DE PÁGINA
# =====================================================
st.set_page_config(
    page_title="EndoPredict Pro",
    page_icon="⚕️",
    layout="wide"
)



# =====================================================
# ESTADO DE NAVEGACIÓN
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "modelo"

# =====================================================
# BARRA SUPERIOR (BOTONES FUNCIONALES)
# =====================================================

nav = st.columns(4)

def nav_button(label, page_name):
    active = "nav-btn-active" if st.session_state.page == page_name else ""
    if st.button(
        label,
        key=f"nav_{page_name}",
        help=label
    ):
        st.session_state.page = page_name
    st.markdown(
        f"""
        <script>
        var btn = window.parent.document.querySelector(
            'button[data-testid="baseButton-secondary"][aria-label="{label}"]'
        );
        if (btn) btn.classList.add("nav-btn", "{active}");
        </script>
        """,
        unsafe_allow_html=True
    )

with nav[0]:
    nav_button(" Modelo", "modelo")

with nav[1]:
    nav_button(" Paciente", "paciente")

with nav[2]:
    nav_button(" Metodología", "metodologia")

with nav[3]:
    nav_button(" Contacto", "contacto")

st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# HEADER: LOGO + TÍTULO (DEBAJO DE LA BARRA)
# =====================================================
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("logo_fib.png", width=80)

with col_title:
    st.title("EndoPredict Pro: Evaluación de Riesgo en Cáncer de Endometrio")
    st.markdown("### *Claridad predictiva para decisiones clínicas informadas*")

st.markdown("---")

# =====================================================
# CONTENIDO SEGÚN SECCIÓN
# =====================================================
page = st.session_state.page

# -----------------------------
# 📊 MODELO
# -----------------------------
if page == "modelo":
    st.header("📊 Visión General del Modelo")
    st.write("Descripción del modelo y métricas.")
    st.metric("AUC ROC", "0.92")

# -----------------------------
# 🔬 PACIENTE
# -----------------------------
elif page == "paciente":
    st.header("🔬 Evaluación de Pacientes")
    st.write("""
    Sube un archivo CSV con los datos clínicos del paciente.  
    El sistema realizará una predicción simulada:
    - Probabilidad de cáncer  
    - Tipo de cáncer más probable  
    - Factores que más influyen  
    - Recomendaciones personalizadas  
    """)

    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # ---------------- Pestañas principales ----------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Datos Generales",
            "Predicción IA",
            "Recomendaciones",
            "Gráficos"
        ])

        # ---------------- TAB 1: DATOS GENERALES ----------------
        with tab1:
            st.subheader("Datos clínicos del paciente")
            for idx, row in df.iterrows():
                st.markdown(f"### Paciente {idx+1}")
                tabla = pd.DataFrame({
                    "Variable": row.index,
                    "Valor": row.values
                })
                st.table(tabla)
                st.divider()

        # ---------------- TAB 2: PREDICCIÓN IA ----------------
        with tab2:
            st.subheader("Resultados de la IA (simulados)")
            np.random.seed(42)
            df["prob_cancer"] = np.random.uniform(0.05, 0.90, size=len(df))
            tipos = ["Endometrioide", "Seroso", "Carcinosarcoma", "Claros", "Mucinoso"]
            df["tipo_probable"] = np.random.choice(tipos, size=len(df))
            factores = ["Edad", "IMC", "Histología", "Grado", "CA125", "Infiltración", "LVSI"]
            df["factor_principal"] = np.random.choice(factores, size=len(df))

            for idx, row in df.iterrows():
                st.markdown(f"### Paciente {idx+1}")
                st.metric("Probabilidad estimada de cáncer", f"{row['prob_cancer']*100:.1f}%")
                st.write("**Tipo más probable:**", row["tipo_probable"])
                st.write("**Factor clínico más influyente:**", row["factor_principal"])
                st.divider()

        # ---------------- TAB 3: RECOMENDACIONES ----------------
        with tab3:
            st.subheader("Recomendaciones clínicas personalizadas")
            for idx, row in df.iterrows():
                st.markdown(f"### Paciente {idx+1}")
                recomendaciones = []

                # Riesgo global
                if row["prob_cancer"] > 0.60:
                    recomendaciones.append("🔴 **Alta probabilidad de cáncer:** Derivación a especialista + pruebas de imagen.")
                elif row["prob_cancer"] > 0.30:
                    recomendaciones.append("🟠 **Probabilidad moderada:** Estudio complementario y seguimiento más frecuente.")
                else:
                    recomendaciones.append("🟢 **Riesgo bajo:** Mantener controles habituales y estilo de vida saludable.")

                # Factor principal
                if row["factor_principal"] == "IMC":
                    recomendaciones.append("➤ IMC influyente: valoración nutricional recomendada.")
                if row["factor_principal"] == "Edad":
                    recomendaciones.append("➤ Edad influyente: seguimiento más frecuente.")
                if row["factor_principal"] == "CA125":
                    recomendaciones.append("➤ Repetir marcadores tumorales y valorar imagen.")
                if row["factor_principal"] == "Infiltración":
                    recomendaciones.append("➤ Puede ser útil resonancia o TAC.")
                if row["factor_principal"] == "LVSI":
                    recomendaciones.append("➤ Valorar afectación ganglionar.")

                # IMC detallado
                if "IMC" in row.index and not pd.isna(row["IMC"]):
                    imc = row["IMC"]
                    recomendaciones.append(f"📊 **IMC del paciente:** {imc:.1f}")
                    if imc < 18.5:
                        recomendaciones.append("⚠️ IMC bajo: aumentar peso de forma controlada.")
                    elif 18.5 <= imc < 25:
                        recomendaciones.append("✔ IMC saludable: mantener estilo de vida y actividad física.")
                    elif 25 <= imc < 30:
                        recomendaciones.append("📉 Sobrepeso: reducir 5–10% del peso.")
                    elif 30 <= imc < 35:
                        recomendaciones.append("📉 Obesidad I: pérdida 10–15% del peso.")
                    elif 35 <= imc < 40:
                        recomendaciones.append("🔴 Obesidad II: pérdida supervisada y seguimiento endocrinología.")
                    else:
                        recomendaciones.append("🚨 Obesidad mórbida: intervención especializada.")

                for r in recomendaciones:
                    st.write(r)
                st.divider()

        # ---------------- TAB 4: GRÁFICOS ----------------
        with tab4:
            st.subheader("Visualización del Riesgo")
            st.bar_chart(df["prob_cancer"])
            st.write("Distribución de tipos más probables")
            st.bar_chart(df["tipo_probable"].value_counts())

    else:
        st.info("Sube un archivo CSV para comenzar.")



# -----------------------------
# 📚 METODOLOGÍA
# -----------------------------
elif page == "metodologia":
    st.header("📚 Metodología")
    st.write("Detalles técnicos y limitaciones.")

# -----------------------------
# 📧 CONTACTO
# -----------------------------
elif page == "contacto":
    st.header("📧 Contacto")
    st.write("Información de contacto.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown("© 2025 EndoPredict Pro")
