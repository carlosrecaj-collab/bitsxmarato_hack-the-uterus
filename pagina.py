import streamlit as st
import pandas as pd
import numpy as np

# -------------------------
# Título de la app
# -------------------------
st.title("Predictor de Cáncer")
st.write("""
Esta aplicación predice la probabilidad de cáncer según los datos ingresados.
Se trata de un ejemplo básico para mostrar resultados de una IA.
""")

# -------------------------
# Subida de datos
# -------------------------
st.header("Sube tus datos")
uploaded_file = st.file_uploader("Elige un archivo CSV con los datos", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(df)
else:
    st.write("O ingresa manualmente los datos:")

    # Ejemplo de inputs manuales
    edad = st.number_input("Edad", min_value=0, max_value=120, value=50)
    tamano_tumor = st.number_input("Tamaño del tumor (mm)", min_value=0.0, max_value=100.0, value=20.0)
    nodos = st.number_input("Número de ganglios afectados", min_value=0, max_value=50, value=2)

    # Crear un dataframe con los datos manuales
    df = pd.DataFrame({
        "Edad": [edad],
        "Tamaño Tumor": [tamano_tumor],
        "Nodos": [nodos]
    })

# -------------------------
# Predicción (simulada)
# -------------------------
st.header("Predicción")

if st.button("Predecir"):
    # Aquí iría tu modelo de IA real
    # Ejemplo con valores aleatorios
    df["Probabilidad Cáncer (%)"] = np.random.uniform(0, 100, size=len(df))

    st.write("Resultados de la predicción:")
    st.dataframe(df)

    # Gráfico sencillo
    st.bar_chart(df["Probabilidad Cáncer (%)"])
