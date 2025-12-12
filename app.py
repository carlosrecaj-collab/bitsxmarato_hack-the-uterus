import streamlit as st
import pandas as pd

st.title("Hola Yago!")
st.write("Tu primera app con Streamlit en Ubuntu :)")


st.title("Título grande")
st.header("Header")
st.subheader("Subheader")
st.write("Texto normal o variables")
st.markdown("**Markdown** _también funciona_")

st.dataframe(df)      # Tabla interactiva
st.table(df)          # Tabla estática
st.json(objeto_json)  # Mostrar JSON

nombre = st.text_input("Escribe tu nombre")

edad = st.number_input("Edad", min_value=0, max_value=120)
valor = st.slider("Selecciona un valor", 0, 100)

opcion = st.selectbox("Elige una opción", ["A", "B", "C"])

if st.button("Pulsar"):
    st.write("¡Botón pulsado!")

archivo = st.file_uploader("Sube una imagen", type=["png", "jpg"])

from PIL import Image
img = Image.open("foto.png")
st.image(img, caption="Mi imagen", use_column_width=True)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3], [10,20,30])
st.pyplot(fig)

col1, col2 = st.columns(2)
col1.write("Columna 1")
col2.write("Columna 2")

op = st.sidebar.selectbox("Menú", ["Home", "Ajustes"])

if "contador" not in st.session_state:
    st.session_state.contador = 0

if st.button("Sumar"):
    st.session_state.contador += 1

st.write("Contador:", st.session_state.contador)

st.success("Todo OK")
st.error("Error")
st.warning("Aviso")
st.info("Info")

st.code("print('hola')", language="python")
