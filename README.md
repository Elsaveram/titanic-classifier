# Proyecto de Clasificación del Titanic

Este repositorio contiene un pipeline de clasificación desarrollado para el dataset del Titanic, como parte de una prueba técnica para construir un proyecto de machine learning "end to end". El objetivo es realizar un análisis exploratorio de datos (EDA), preprocesamiento, selección de modelos, evaluación del modelo seleccionado y despliegue del modelo seleccionado en producción. El pipeline final y los scripts están diseñados para ser fácilmente desplegados usando Docker.

Puedes leer más detalles sobre la implementación de este proyecto en mi [blog post](https://nycdatascience.com/blog/student-works/building-a-titanic-classifier-with-end-to-end-machine-learning-pipeline/).

## Dataset

El dataset utilizado en este proyecto es el del Titanic, que se puede encontrar [aquí en Kaggle](https://www.kaggle.com/c/titanic/data?select=gender_submission.csv). El dataset proporciona información sobre los pasajeros a bordo del Titanic e incluye las siguientes características:

- **PassengerId**: Identificador único para cada pasajero.
- **Survived**: Supervivencia (0 = No, 1 = Sí).
- **Pclass**: Clase del billete (1 = 1ra, 2 = 2da, 3 = 3ra).
- **Name**: Nombre del pasajero.
- **Sex**: Género del pasajero.
- **Age**: Edad del pasajero en años.
- **SibSp**: Número de hermanos/esposos a bordo del Titanic.
- **Parch**: Número de padres/hijos a bordo del Titanic.
- **Ticket**: Número de billete.
- **Fare**: Tarifa del pasajero.
- **Cabin**: Número de cabina.
- **Embarked**: Puerto de embarque (C = Cherburgo, Q = Queenstown, S = Southampton).

## Ideation Notebook

La exploración inicial de datos, el feature engineering, la selección y la evaluación de modelos se detallan en el [Notebook de Ideación del Titanic](https://github.com/Elsaveram/titanic-classifier/blob/main/notebooks/Titanic_ideation.ipynb). Los pasos clave realizados en este notebook son:

- **Análisis Exploratorio de Datos (EDA)**: Realización de un análisis exploratorio para identificar patrones, distribuciones y missing values en las variables del dataset.
- **Missing Data Imputation**: Imputación de datos faltantes, particularmente en las columnas `Age` y `Cabin`.
- **Transformación y Feature Engineering**: Creación de nuevos features a partir de los existentes y transformación de variables categóricas, continuas y ordinales. Generación de un archivo `scaler.pkl` que se usará en el proceso de inferencia en producción.
- **Train/Test Split**: División del dataset en train y test set para evaluar el rendimiento de los modelos de manera robusta.
- **Optimización de Hiperparámetros**: Uso de la técnica de Random Search para ajustar los hiperparámetros del modelo y mejorar su rendimiento.
- **Selección de Características (Feature Selection)**: Se ha utilizado la importancia de los features de Random Forest y el estudio de correlaciones para seleccionar las características más relevantes y mimimizar el multicolinearity. Las características seleccionadas como finales para el modelo de inferencia en producción son: `Pclass`, `Age`, `FamilySize`, `Sex_male`, `Title_encoded`, `Deck_encoded`, `Fare_log`.
- **Selección de Modelos**: Prueba de diferentes modelos de clasificación. Se han comparado en particular los modelos de Logistic Regression, Random Forest y XGBoost.
- **Evaluación de Modelos**: Selección del mejor modelo basado en métricas como la accuracy, precisión, el recall, y otras métricas relevantes. La selección del mejor modelo se ha realizado en función de los performance metrics del train set. Se ha seleccionado el modelo Random Forest como modelo final para usar en el proceso de inferencia en producción. Una vez hecho esto, se ha calculado el mejor threshold en base al gráfico ROC AUC y se ha calculado la performance en el test set. Tanto el modelo optimizado (`titanic_RandomForestClassifier_full.pkl`) como el mejor threshold (`config file`) se han almacenado para su uso en producción.

## Pipeline y Scripts

El pipeline principal de producción está ubicado en el [Pipeline del Titanic](https://github.com/Elsaveram/titanic-classifier/blob/main/src/titanic_pipeline.py). Este script importa el [script de preprocesamiento](https://github.com/Elsaveram/titanic-classifier/blob/main/src/titanic_preprocessing.py) y preprocesa los datos, aplica el modelo seleccionado a los datos preprocesados, importa el [script de predicción](https://github.com/Elsaveram/titanic-classifier/blob/main/src/titanic_forecast.py) y genera predicciones que almacena en la tabla output final. Está diseñado para un despliegue fácil y una integración en un entorno de producción usando Docker.

### Pruebas Unitarias

Se incluyen pruebas unitarias en el repositorio como muestra y marcador de posición. Idealmente, cada función en el pipeline debería tener una prueba unitaria correspondiente para asegurar su corrección. Estas pruebas están ubicadas en la carpeta `tests`.

## Ejecución del Proyecto

Este proyecto está containerizado usando Docker, y puedes ejecutar fácilmente los notebooks, el pipeline y las pruebas siguiendo estas instrucciones:

### Requisitos Previos

- Asegúrate de que Docker Desktop esté instalado en tu ordenador.

Para ejecutar los notebooks y los scripts, sigue las instrucciones proporcionadas en las secciones correspondientes.

### Cómo iniciar el contenedor de Notebook

1. Inicia el contenedor de Jupyter Notebook ejecutando:

```sh
docker compose up jupyter
```

2. Después de que el contenedor se inicie, haz clic en el enlace 127.0.0.1 que aparece en la terminal para abrir la interfaz de Jupyter Notebook en tu navegador.

### Cómo ejecutar el pipeline y las pruebas

Para ejecutar el pipeline de Titanic y las pruebas asociadas, utiliza el siguiente comando:

```sh
docker compose run ops ./scripts/run_titanic_pipeline_and_tests.sh
```
Este comando ejecutará el script del pipeline y correrá las pruebas unitarias de ejemplo incluidas en el repositorio.

## Conclusión

Este proyecto demuestra el proceso completo de construcción y despliegue de un modelo de machine learning utilizando el dataset del Titanic. Incluye todo, desde el análisis exploratorio hasta los scripts de despliegue final, asegurando que el modelo se pueda desplegar sin problemas en entornos de producción. El repositorio está estructurado de manera modular, con pruebas unitarias y scripts que pueden extenderse fácilmente para futuras mejoras.


