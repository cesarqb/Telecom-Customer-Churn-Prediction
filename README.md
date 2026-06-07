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

Técnicas utilizadas

Durante el desarrollo del proyecto se aplicaron diversas técnicas de Ciencia de Datos y Machine Learning:

Preparación de datos
Limpieza de datos
Tratamiento de valores faltantes
Conversión de variables categóricas
Codificación mediante One-Hot Encoding
Escalado de variables numéricas
Ingeniería de características
Análisis Exploratorio de Datos (EDA)
Distribución de variables
Análisis univariado
Análisis bivariado
Análisis de correlación
Visualización de patrones de churn
Evaluación del desbalance de clases
Balanceo de clases

Debido al desbalance existente entre clientes que abandonan y los que permanecen, se aplicaron diferentes estrategias:

Oversampling
Undersampling
SMOTE (Synthetic Minority Oversampling Technique)
Modelos de Machine Learning

Se entrenaron y compararon diferentes modelos de clasificación:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier
Optimización de modelos
GridSearchCV
Cross Validation
Evaluación de modelos
Accuracy
Precision
Recall
F1 Score
ROC Curve
ROC-AUC
Confusion Matrix
Classification Report
Persistencia del modelo
Pickle
Joblib
Tecnologías utilizadas
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
XGBoost
Imbalanced-Learn (SMOTE)
SciPy
Pickle
Joblib
Estructura del proyecto

Telecom-Customer-Churn-Prediction

│

├── README.md

├── requirements.txt

├── churn_notebook.ipynb

│

├── data

│ ├── raw

│ │ └── telecom_churn.csv

│ │

│ └── processed

│ └── telecom_churn_clean.csv

│

├── notebooks

│ └── churn_modeling.ipynb

│

├── src

│ ├── preprocessing.py

│ ├── feature_engineering.py

│ ├── balancing.py

│ ├── eda_evaluation.py

│ ├── train_model.py

│ └── evaluation.py

│

├── images

│

├── models

│ └── churn_model.pkl

│

└── results

├── metrics.csv

└── predictions.csv

Descripción de carpetas
data/

Contiene los datos originales y los conjuntos de datos procesados utilizados durante el modelado.

notebooks/

Incluye notebooks utilizados para el análisis exploratorio, preparación de datos y construcción de modelos.

src/

Contiene funciones reutilizables para:

Preprocesamiento
Ingeniería de características
Balanceo de clases
Entrenamiento de modelos
Evaluación de resultados
images/

Almacena gráficos y visualizaciones generadas durante el análisis.

models/

Contiene los modelos entrenados serializados para futuras predicciones.

results/

Incluye métricas de evaluación y resultados finales de predicción.

Flujo del análisis

El proyecto sigue las siguientes etapas:

Carga y exploración inicial de datos.
Análisis de calidad de datos.
Tratamiento de valores faltantes.
Análisis exploratorio de datos (EDA).
Estudio del comportamiento de churn.
Transformación de variables categóricas.
Codificación mediante One-Hot Encoding.
Escalado de variables.
Análisis del desbalance de clases.
Aplicación de técnicas de remuestreo:
Oversampling
Undersampling
SMOTE
División Train/Test.
Entrenamiento de modelos de clasificación.
Optimización de hiperparámetros mediante GridSearchCV.
Evaluación comparativa de modelos.
Análisis de métricas de desempeño.
Interpretación de resultados.
Selección del mejor modelo.
Guardado del modelo entrenado.
Generación de predicciones para nuevos clientes.
Ejemplo de resultado

Cliente:

Tipo de contrato: Mensual
Antigüedad: 8 meses
Servicio de internet: Fibra óptica
Método de pago: Electronic Check

Resultado del modelo:

Probabilidad de abandono: 87%

Predicción:

CHURN = YES

Variables analizadas

Algunas variables incluidas en el análisis:

Gender
Senior Citizen
Partner
Dependents
Tenure
Phone Service
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
Contract
Paperless Billing
Payment Method
Monthly Charges
Total Charges
Impacto de negocio

La implementación de un modelo de predicción de churn permite:

Identificar clientes en riesgo antes de que abandonen la empresa.
Diseñar campañas de retención personalizadas.
Optimizar recursos comerciales.
Reducir costos de adquisición de nuevos clientes.
Mejorar la satisfacción del cliente.
Incrementar el Customer Lifetime Value (CLV).
Posibles mejoras futuras
Implementar LightGBM y CatBoost.
Aplicar técnicas avanzadas de Feature Selection.
Incorporar SHAP para interpretabilidad del modelo.
Automatizar pipelines mediante Scikit-Learn Pipeline.
Implementar seguimiento de experimentos con MLflow.
Desplegar el modelo mediante FastAPI.
Crear un dashboard interactivo con Streamlit.
Integrar monitoreo de desempeño en producción.
Resultados esperados

El proyecto busca obtener un modelo capaz de identificar clientes con alto riesgo de abandono mediante métricas robustas de clasificación, priorizando especialmente:

Recall
F1 Score
ROC-AUC

Estas métricas son fundamentales en problemas de churn debido al impacto económico asociado a la pérdida de clientes.

Autor

César Quezada

Científico de Datos | Docente Universitario | Mentor

Especialista en Machine Learning, Analítica de Datos e Inteligencia Artificial.
