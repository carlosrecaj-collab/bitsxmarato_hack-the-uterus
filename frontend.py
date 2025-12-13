import streamlit as st
import pandas as pd
import numpy as np
import requests # Para conectar con tu API de backend
import json # Para manejar JSON

# --- Configuración de la página ---
st.set_page_config(
    page_title="EndoPredict Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- URL de tu API de Backend ---
BACKEND_URL = "http://tu_ip_o_dominio:puerto/predict" # ¡CAMBIA ESTO!
SHAP_URL = "http://tu_ip_o_dominio:puerto/explain" # ¡CAMBIA ESTO si tienes endpoint SHAP!

# --- Logo y Título Principal ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://via.placeholder.com/100x100?text=EndoLogo", width=80) # Reemplaza con tu logo real
with col_title:
    st.title("EndoPredict Pro: Evaluación de Riesgo en Cáncer de Endometrio")
    st.markdown("### *Claridad predictiva para decisiones clínicas informadas*")

st.markdown("---")

# --- Sidebar para Navegación ---
st.sidebar.title("Navegación")
selection = st.sidebar.radio(
    "Ir a:",
    ["📊 Visión General del Modelo", "🔬 Evaluación de Pacientes", "📚 Recursos y Metodología", "📧 Contacto y Soporte"]
)

# --- Contenido de las Páginas ---

if selection == "📊 Visión General del Modelo":
    st.header("📊 Visión General del Modelo")
    st.write("Explora la influencia de cada variable en las predicciones de recaída.")

    st.markdown("#### **Fuerza Predictiva Dinámica (Concepto SHAP simplificado)**")
    st.write("Ajusta los sliders para ver cómo diferentes características afectan la importancia relativa en una predicción hipotética.")

    # Ejemplo interactivo de pesos de variables
    st.subheader("Variables Clave y su Impacto Teórico")
    col1, col2 = st.columns(2)

    with col1:
        age_impact = st.slider("Edad del Paciente", 30, 90, 60, help="Impacto de la edad en el riesgo.")
        grade_impact = st.selectbox("Grado Histológico", ["G1", "G2", "G3"], help="Impacto del grado tumoral.")
        # Agrega más sliders/selectores para tus variables
        st.info(f"Con una edad de {age_impact} años y grado {grade_impact}, el riesgo teórico se ajusta en un X%.")


    # Aquí iría un gráfico interactivo (ej. Radar, Waterfall, o SHAP resumen)
    st.warning("👉 *Este sería el lugar ideal para un gráfico de 'sol de influencia' o un SHAP summary plot interactivo, mostrando pesos relativos.*")
    st.markdown("---")
    st.subheader("Métricas de Rendimiento del Modelo")
    st.metric(label="Área bajo la Curva ROC (AUC)", value="0.92", delta="Excelente precisión")
    st.write("Estas métricas demuestran la robustez de nuestro modelo en la validación.")
    # Aquí un gráfico de curva ROC simplificado
    st.image("https://via.placeholder.com/400x200?text=Curva+ROC", caption="Curva ROC del modelo", use_column_width=True) # Reemplaza con tu gráfico real


elif selection == "🔬 Evaluación de Pacientes":
    st.header("🔬 Evaluación de Pacientes")
    st.write("Introduce los parámetros del paciente para obtener una predicción personalizada del riesgo de recaída.")

    # --- Formulario de Entrada de Datos ---
    with st.form("patient_data_form"):
        st.subheader("Datos Demográficos y Clínicos")
        col_dem1, col_dem2 = st.columns(2)
        with col_dem1:
            edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=65)
            # Ejemplo de validación simple
            if edad < 18 or edad > 100:
                st.error("Por favor, introduce una edad válida.")
            peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
            altura = st.number_input("Altura (cm)", min_value=100, max_value=250, value=165)
        with col_dem2:
            menopausia = st.selectbox("Estado Menopáusico", ["Pre-menopáusica", "Post-menopáusica", "Perimenopáusica"])
            diabetes = st.checkbox("Diabetes Presente")
            hipertension = st.checkbox("Hipertensión Presente")
            # Agrega más inputs según tus variables

        st.subheader("Hallazgos Histopatológicos y Biomoleculares")
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            grado_tumoral = st.radio("Grado Histológico", ["G1", "G2", "G3"])
            tipo_histologico = st.selectbox("Tipo Histológico", ["Endometriode", "Seroso", "Células Claras", "Otros"])
        with col_hist2:
            invasion_linfovascular = st.checkbox("Invasión Linfovascular Presente")
            mutacion_p53 = st.checkbox("Mutación TP53")
            # Agrega más inputs según tus variables

        # --- Botón de Envío ---
        submitted = st.form_submit_button("Calcular Riesgo de Recaída")

    if submitted:
        st.info("Calculando la probabilidad de recaída...")

        # Preparar los datos para enviar al backend
        patient_data = {
            "edad": edad,
            "peso": peso,
            "altura": altura,
            "menopausia": menopausia,
            "diabetes": diabetes,
            "hipertension": hipertension,
            "grado_tumoral": grado_tumoral,
            "tipo_histologico": tipo_histologico,
            "invasion_linfovascular": invasion_linfovascular,
            "mutacion_p53": mutacion_p53,
            # ... todas tus variables del modelo
        }

        try:
            # Enviar datos al backend para predicción
            response = requests.post(BACKEND_URL, json=patient_data)
            response.raise_for_status() # Lanza un error para códigos de estado HTTP incorrectos
            prediction_result = response.json()

            # --- Mostrar Resultados ---
            st.subheader("Resultados de la Predicción")
            prob_recaida = prediction_result.get("probabilidad_recaida", 0.0) # Ajusta la clave según tu API

            # Medidor de Riesgo Dinámico
            st.markdown(f"**Probabilidad de Recaída:** `{prob_recaida:.2%}`")
            if prob_recaida < 0.2:
                st.success("Riesgo Bajo de Recaída")
            elif prob_recaida < 0.5:
                st.warning("Riesgo Moderado de Recaída")
            else:
                st.error("Riesgo Alto de Recaída")

            # Idea: Gráfico de "Explicación Localizada" (SHAP/LIME simplificado)
            st.markdown("#### Factores que Influyen en esta Predicción Específica")
            # Aquí se llamarías a otro endpoint de tu API para obtener los valores SHAP o LIME
            # Por simplicidad, un ejemplo dummy:
            if SHAP_URL:
                 try:
                    shap_response = requests.post(SHAP_URL, json=patient_data)
                    shap_response.raise_for_status()
                    shap_values = shap_response.json().get("shap_contributions", {}) # Ajusta la clave
                    
                    if shap_values:
                        st.write("Estos son los factores que más influyen en la probabilidad de recaída de este paciente:")
                        shap_df = pd.DataFrame(list(shap_values.items()), columns=['Característica', 'Impacto'])
                        shap_df['Color'] = shap_df['Impacto'].apply(lambda x: 'red' if x > 0 else 'green')
                        # st.bar_chart(shap_df.set_index('Característica')['Impacto']) # Una forma sencilla
                        # Una visualización más sofisticada podría requerir librerías como altair o plotly
                        for index, row in shap_df.iterrows():
                            color = "red" if row['Impacto'] > 0 else "green"
                            sign = "+" if row['Impacto'] > 0 else ""
                            st.markdown(f"- **{row['Característica']}**: <span style='color:{color}'>{sign}{row['Impacto']:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.write("No se pudo obtener la explicación de los factores en este momento.")

                 except requests.exceptions.RequestException as e:
                    st.error(f"Error al obtener la explicación del modelo: {e}")
            else:
                st.info("La explicación de los factores está en desarrollo.")


            # Botón de descarga de reporte (Placeholder)
            st.download_button(
                label="Descargar Reporte PDF",
                data="Datos del reporte", # Aquí irían los datos reales del PDF
                file_name="Reporte_Paciente_EndoPredict.pdf",
                mime="application/pdf"
            )

        except requests.exceptions.ConnectionError:
            st.error("Error: No se pudo conectar con el servidor del modelo de IA. Por favor, asegúrate de que el backend esté funcionando.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Error HTTP del servidor: {e}. Por favor, verifica los datos enviados.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar la predicción: {e}")

elif selection == "📚 Recursos y Metodología":
    st.header("📚 Recursos y Metodología")
    st.write("Aquí encontrarás información detallada sobre el modelo de IA, los datos utilizados y la metodología.")
    st.subheader("Nuestro Modelo de IA")
    st.markdown("""
    Nuestro modelo utiliza un algoritmo de **XGBoost (eXtreme Gradient Boosting)**, entrenado para identificar patrones complejos en datos de pacientes con cáncer de endometrio. Este tipo de modelo es conocido por su alta precisión y capacidad para manejar diversos tipos de datos.

    **Ventajas:**
    * Alta precisión en la predicción.
    * Robustez frente a datos faltantes o ruidosos.
    * Capacidad para identificar interacciones complejas entre variables.
    """)
    st.subheader("Conjunto de Datos")
    st.write("El modelo fue entrenado con un conjunto de datos anonimizado de X pacientes, recopilado de Y instituciones colaboradoras. El dataset incluye variables demográficas, histopatológicas y biomoleculares.")
    st.markdown("---")
    st.subheader("Limitaciones del Modelo")
    st.warning("""
    * **No es un reemplazo para el juicio clínico:** Este modelo es una herramienta de apoyo y no debe sustituir la evaluación y decisión de un profesional médico.
    * **Dependencia de los datos de entrenamiento:** La precisión del modelo puede variar en poblaciones o escenarios que difieran significativamente de los datos utilizados para su entrenamiento.
    * **Faltan algunos marcadores emergentes:** Aunque el modelo es robusto, la investigación en cáncer de endometrio está en constante evolución.
    """)
    st.subheader("Bibliografía y Referencias")
    st.markdown("""
    * [Artículo 1: Deep Learning for Endometrial Cancer Prognosis](https://example.com/article1)
    * [Artículo 2: XGBoost in Medical Prediction](https://example.com/article2)
    """)

elif selection == "📧 Contacto y Soporte":
    st.header("📧 Contacto y Soporte")
    st.write("Para cualquier consulta, sugerencia o soporte técnico, por favor, contacta con nosotros.")
    st.markdown("""
    **Equipo de Desarrollo:** [Tu Nombre/Equipo]
    **Correo Electrónico:** [tu.correo@ejemplo.com]
    **Enlace de LinkedIn:** [Tu perfil de LinkedIn]
    """)
    st.markdown("---")
    st.subheader("Envíanos tus Comentarios")
    with st.form("feedback_form"):
        nombre_feedback = st.text_input("Tu Nombre (Opcional)")
        email_feedback = st.text_input("Tu Correo Electrónico (Opcional)")
        mensaje_feedback = st.text_area("Tu Mensaje", height=150)
        submitted_feedback = st.form_submit_button("Enviar Comentarios")
        if submitted_feedback:
            st.success("¡Gracias por tus comentarios! Nos pondremos en contacto si es necesario.")
            # Aquí podrías integrar un servicio para enviar estos comentarios por email o guardarlos.

# --- Pie de página ---
st.markdown("---")
st.markdown("© 2025. Desarrollado para la evaluación de riesgo en cáncer de endometrio.")