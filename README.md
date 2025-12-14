# EndoPredict

**Equipo:** Ignasi, Pau, Carlos y Yago

**EndoPredict** es una aplicación diseñada para la predicción de riesgo relacionada con cáncer de endometrio. Nuestro objetivo ha sido combinar técnicas de análisis de datos y machine learning para ofrecer una alta probabilidad de detección temprana.

---

## Tecnologías y metodología

- **Modelo de Machine Learning:** Se ha empleado un enfoque combinado de **clustering + distancia escalar** antes de entrenar un **modelo XGBoost**, logrando una probabilidad de detección elevada.
- **Front-end:** La interfaz de usuario está desarrollada con **Streamlit**, facilitando la interacción intuitiva y visual.
- **Nombre de la aplicación:** `EndoPredict`

---

## Uso

1. Clonar este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/endopredict.git
2. Crear un entorno virtual e instalar las dependencias:
    ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
3. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
