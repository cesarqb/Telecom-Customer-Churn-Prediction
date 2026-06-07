# TELECOM CUSTOMER CHURN PREDICTION WITH MACHINE LEARNING

Proyecto completo de Ciencia de Datos para la predicción de fuga de clientes (Customer Churn) en una empresa de telecomunicaciones utilizando técnicas de análisis exploratorio de datos, balanceo de clases y modelos de Machine Learning.

---

## Objetivos del proyecto
- Identificar clientes con alta probabilidad de abandonar el servicio para apoyar la toma de decisiones estratégicas orientadas a la retención de clientes y reducción de pérdidas económicas.
- Analizar el comportamiento de los clientes de una empresa de telecomunicaciones.
- Identificar los factores asociados al abandono del servicio.
- Manejar el desbalance de clases mediante técnicas de remuestreo.
- Construir modelos predictivos de clasificación.
- Optimizar hiperparámetros mediante GridSearchCV.
- Identificar las variables con mayor influencia en la fuga de clientes.
- Serializar el modelo para futuras implementaciones.

--- 

## Técnicas utilizadas

Durante el desarrollo del proyecto se aplicaron diversas técnicas de Ciencia de Datos y Machine Learning:

- Preparación de datos
- Limpieza de datos
- Tratamiento de valores faltantes
- Conversión de variables categóricas
- Codificación mediante One-Hot Encoding
- Escalado de variables numéricas
- Ingeniería de características
- Análisis Exploratorio de Datos (EDA)
- Análisis univariado, bivariado y multivariado
- Análisis de correlación
- Visualización de patrones de churn
- Evaluación del desbalance de clases
- Balanceo de clases
- Modelos utilizados:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - XGBoost Classifier
- Optimización de modelos
- Serialización del modelo (pickle)

--- 

## Tecnologías utilizadas

- Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-Learn (SMOTE), SciPy, Pickle, Joblib

---

## Estructura del proyecto

```
Telecom-Customer-Churn-Prediction
│
├── README.md
├── requirements.txt
├── churn_notebook.ipynb
│
├── data
│   ├── raw
│   │   └── telecom_churn.csv
│   │
│   └── processed
│       └── telecom_churn_clean.csv
│
├── notebooks
│   └── churn_modeling.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── balancing.py
│   ├── eda_evaluation.py
│   ├── train_model.py
│   └── evaluation.py
│
├── images
│
├── models
│   └── churn_model.pkl
│
└── results
    ├── metrics.csv
    └── predictions.csv

```
---
## Descripción de carpetas

* data/ → contiene datos originales (vivienda, vivienda_new) y procesados.
* notebooks/ → análisis exploratorio y desarrollo del modelo.
* src/ → funciones reutilizables (preprocesamiento, modelado, evaluación).
* images/ → visualizaciones generadas durante el análisis.
* models/ → modelo entrenado guardado con pickle.
* results/ → predicciones y métricas finales.

---

## Flujo del análisis

El análisis sigue las siguientes etapas:

1. Carga y exploración inicial de datos.
2. Análisis de calidad de datos.
3. Tratamiento de valores faltantes.
4. Análisis exploratorio de datos (EDA).
5. Estudio del comportamiento de churn.
6. Transformación de variables categóricas.
7. Codificación mediante One-Hot Encoding.
8. Escalado de variables.
9. Análisis del desbalance de clases.
10. Aplicación de técnicas de remuestreo.
11. División Train/Test.
12. Entrenamiento de modelos de clasificación.
13. Optimización de hiperparámetros mediante GridSearchCV.
14. Evaluación comparativa de modelos.
15. Análisis de métricas de desempeño.
16. Guardado del modelo entrenado.
17. Generación de predicciones para nuevos clientes.

---

## Ejemplo de resultado

Cliente:

Tipo de contrato: Mensual
Antigüedad: 8 meses
Servicio de internet: Fibra óptica
Método de pago: Electronic Check

Resultado del modelo: Probabilidad de abandono 87%

Predicción: CHURN = SI

--- 

## Posibles mejoras futuras

- Implementar modelos más avanzados como LightGBM y CatBoost.
- Automatizar el flujo de trabajo con Scikit-learn Pipeline.
- Aplicar técnicas más robustas de selección de variables.
- Desplegar el modelo mediante FastAPI o Streamlit.

---

## 👨‍🏫 Autor

César Quezada
Científico de Datos | Docente Universitario | Mentor
