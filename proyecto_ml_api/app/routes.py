# routes.py
from uuid import uuid4
from bson import ObjectId
from flask import Blueprint, request, jsonify
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import config
from dbconection import connect_to_mongo
import pandas as pd
import json

# Cargar la configuración desde el archivo
with open('config.json') as config_file:
    config = json.load(config_file)


# Crear un Blueprint para las rutas
routes_bp = Blueprint('routes', __name__)

# Obtener la conexión a MongoDB
db = connect_to_mongo(config)

@routes_bp.route('/load', methods=['POST'])
def load_excel():
    try:
        # Verificar si se proporcionó un archivo en la solicitud
        if 'file' not in request.files:
            return jsonify({"error": "No se proporcionó ningún archivo"}), 400

        file = request.files['file']

        # Verificar si se seleccionó un archivo
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        # Verificar la extensión del archivo
        file_extension = file.filename.split('.')[-1].lower()

        if file_extension == 'xlsx':
            # Leer el archivo Excel en un DataFrame de pandas
            df = pd.read_excel(file, engine='openpyxl')
        elif file_extension == 'csv':
            # Leer el archivo CSV en un DataFrame de pandas
            df = pd.read_csv(file, sep=";")
        else:
            return jsonify({"error": "Se esperaba un archivo con extensión .xlsx o .csv"}), 400

        # Convertir cada fila del DataFrame a un documento JSON

        datos = {
            'informacion':df.to_dict(orient='records')
        }
        #documents = json.loads(df.to_json(orient='records'))
        # Insertar cada documento en la base de datos
        db.mi_coleccion.insert_one(datos)

        return jsonify({"success": "Documentos cargados exitosamente"}), 200

    except Exception as e:
        return jsonify({"error": f"Error al cargar el archivo: {str(e)}"}), 500

"""
_____________________________________________________________________________________________________
"""

@routes_bp.route('/basic-statistics/<string:dataset_id>', methods=['GET'])
def basic_statistics(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "mi_coleccion"
        document = db[original_collection_name].find_one({"_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer la información del array en el documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        # Obtener estadísticas básicas utilizando el comando describe de pandas
        statistics = original_dataset.describe()

        # Convertir el resultado a un diccionario para la respuesta JSON
        statistics_dict = statistics.to_dict()

        # Convertir el ObjectId a cadena para ser serializado en JSON
        #statistics_dict["_id"] = str(dataset_id_objectid)

        return jsonify({"statistics": statistics_dict})

    except Exception as e:
        return jsonify({"error": f"Error al obtener las estadísticas básicas: {str(e)}"}), 500

"""
_______________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________

"""
@routes_bp.route('/columns-describe/<string:dataset_id>', methods=['GET'])
def columns_describe(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "mi_coleccion"
        document = db[original_collection_name].find_one({"_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        # Obtener la descripción de los tipos de datos de las columnas
        columns_description = get_columns_description(original_dataset)

        return jsonify({"columns_description": columns_description})

    except Exception as e:
        return jsonify({"error": f"Error al obtener la descripción de columnas: {str(e)}"}), 500

def get_columns_description(dataset):
    columns_description = {}

    for column, dtype in dataset.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            columns_description[column] = "Numérico"
        elif pd.api.types.is_string_dtype(dtype):
            columns_description[column] = "Texto"
        else:
            columns_description[column] = "Desconocido"

    return columns_description



"""


@routes_bp.route('/dataset-info', methods=['GET'])
def dataset_info():
    try:
        # Obtener todos los datos de la colección
        data = db.mi_coleccion.find()

        # Convertir los datos a un DataFrame de pandas
        df = pd.DataFrame(data)

        # Obtener información del DataFrame
        info = df.info()

        return jsonify({"dataset_info": info})

    except Exception as e:
        return jsonify({"error": f"Error al procesar los datos: {str(e)}"}), 500
_________________________________________________________________
    
"""

from flask import jsonify, request
import pandas as pd
 # Supongamos que tienes un diccionario que almacena tus conjuntos de datos
datasets = {}

#VERIFICAR BIEN PARA DATOS QUE NO HACEN FALTA IMPUTACION Y ENCONTRAR CUALES SI



@routes_bp.route('/imputation/<string:dataset_id>/type/<int:number_type>', methods=['POST'])
def imputation(dataset_id, number_type):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "mi_coleccion"
        document = db[original_collection_name].find_one({"_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        # Aplicar técnicas de imputación según el tipo seleccionado
        imputed_dataset = apply_imputation(original_dataset, number_type)

        # Crear una copia del dataset original
        copy_collection_name = "Imputación"
        db[copy_collection_name].insert_one({"dataset_id": dataset_id_objectid,"tipo_imputación":number_type, "informacion": imputed_dataset.to_dict(orient='records')})

        return jsonify({"message": "Imputación realizada con éxito. Copia del dataset creada en 'Imputación'"})

    except Exception as e:
        return jsonify({"error": f"Error al realizar la imputación: {str(e)}"}), 500

def apply_imputation(dataset, number_type):
    if number_type == 1:
        # Eliminar registros que contienen datos faltantes
        imputed_dataset = dataset.dropna()
    elif number_type == 2:
        # Imputación por media para variables numéricas, y moda por variables categóricas/texto
        imputed_dataset = dataset.apply(lambda column: column.fillna(column.mean()) if pd.api.types.is_numeric_dtype(column) else column.fillna(column.mode()[0]))
    else:
        raise ValueError("Tipo de imputación no válido")

    return imputed_dataset



"""


def apply_imputation(dataset, number_type):
    dataset_copy = dataset.copy()

    if number_type == 1:
        dataset_copy.dropna(inplace=True)
    elif number_type == 2:
        # Verificar si hay columnas numéricas antes de intentar la imputación
        numeric_columns = dataset_copy.select_dtypes(include=['number']).columns
        if not numeric_columns.empty:
            dataset_copy[numeric_columns] = dataset_copy[numeric_columns].fillna(dataset_copy[numeric_columns].mean())
        else:
            raise ValueError("No hay columnas numéricas para la imputación por media")
    else:
        raise ValueError("Tipo de imputación no válido")

    return dataset_copy
@routes_bp.route('/imputation/<string:dataset_id>/type/<int:number_type>', methods=['POST'])
def imputation(dataset_id, number_type):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)
        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "mi_coleccion"
        original_dataset = pd.DataFrame(list(db[original_collection_name].find({"_id": dataset_id_objectid})))

        if original_dataset.empty:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Aplicar técnicas de imputación según el tipo seleccionado
        imputed_dataset = apply_imputation(original_dataset, int(number_type))

        # Guardar el nuevo conjunto de datos imputado en la colección "imputacion" de MongoDB
        imputed_collection_name = "Imputación"
        db[imputed_collection_name].insert_many(imputed_dataset.to_dict(orient='records'))

        return jsonify({"status": "Imputación exitosa", "imputed_collection_name": imputed_collection_name})

    except Exception as e:
        return jsonify({"error": f"Error al realizar la imputación: {str(e)}"}), 500



________________________________________________________________________________________________

"""
from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


@routes_bp.route('/general-univariate-graphs/<string:dataset_id>', methods=['POST'])
def general_univariate_graphs(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "Imputación"
        document = db[original_collection_name].find_one({"dataset_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        # Crear carpeta con nombre del identificador del dataset para almacenar gráficos
        output_folder = f"graphs_{dataset_id}"
        os.makedirs(output_folder, exist_ok=True)

        # Generar y almacenar gráficos univariados
        generate_univariate_graphs(original_dataset, output_folder)

        return jsonify({"message": "Gráficos univariados generados y almacenados en la carpeta 'graphs_dataset_id'"})

    except Exception as e:
        return jsonify({"error": f"Error al generar gráficos univariados: {str(e)}"}), 500

def generate_univariate_graphs(dataset, output_folder):
    for column in dataset.columns:
        # Histograma
        plt.figure()
        sns.histplot(dataset[column], kde=False)
        plt.title(f'Histograma de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        histogram_path = os.path.join(output_folder, f'histogram_{column}.png')
        plt.savefig(histogram_path)
        plt.close()

        # Diagrama de caja (solo para variables numéricas)
        if pd.api.types.is_numeric_dtype(dataset[column]):
            plt.figure()
            sns.boxplot(x=dataset[column])
            plt.title(f'Diagrama de Caja de {column}')
            plt.xlabel(column)
            plt.ylabel('Valor')
            boxplot_path = os.path.join(output_folder, f'boxplot_{column}.png')
            plt.savefig(boxplot_path)
            plt.close()

        # Análisis de distribución de probabilidad (solo para variables numéricas)
        if pd.api.types.is_numeric_dtype(dataset[column]):
            plt.figure()
            sns.kdeplot(dataset[column], fill=True)
            plt.title(f'Análisis de Distribución de Probabilidad de {column}')
            plt.xlabel(column)
            plt.ylabel('Densidad')
            distribution_path = os.path.join(output_folder, f'distribution_{column}.png')
            plt.savefig(distribution_path)
            plt.close()




"""

______________________________________________________
"""

@routes_bp.route('/univariate-graphs-class/<string:dataset_id>', methods=['POST'])
def univariate_graphs_class(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "Imputación"
        document = db[original_collection_name].find_one({"dataset_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        caca = request.json
        clases_objetivos = caca.get('caca')
        # Filtrar el conjunto de datos para obtener solo las clases 2 y 4
        unique_classes = original_dataset[clases_objetivos].unique()

        filtered_dataset = original_dataset[original_dataset[clases_objetivos].isin(unique_classes)]

        if filtered_dataset.empty:
            return jsonify({"error": "Conjunto de datos sin clases 2 o 4"}), 400

        # Crear carpeta con nombre del identificador del dataset para almacenar gráficos
        output_folder = f"graphs_class_{dataset_id}"
        os.makedirs(output_folder, exist_ok=True)

        # Generar y almacenar gráficos univariados por clase
        generate_univariate_graphs_class(filtered_dataset, output_folder)

        return jsonify({"message": f"Gráficos univariados por clase generados y almacenados en la carpeta 'graphs_class_{dataset_id}'"})

    except Exception as e:
        return jsonify({"error": f"Error al generar gráficos univariados por clase: {str(e)}"}), 500

def generate_univariate_graphs_class(dataset, output_folder):
    caca = request.json
    clases_objetivos = caca.get('caca')
    # Diagrama de caja y gráfico de densidad por clase
    for column in dataset.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=clases_objetivos, y=column, data=dataset)
        plt.title(f'Diagrama de Caja de {column} por Clase')
        plt.xlabel('Clase')
        plt.ylabel(column)
        boxplot_path = os.path.join(output_folder, f'boxplot_class_{column}.png')
        plt.savefig(boxplot_path)
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=column, hue=clases_objetivos, data=dataset, fill=True)
        plt.title(f'Gráfico de Densidad de {column} por Clase')
        plt.xlabel(column)
        plt.ylabel('Densidad')
        density_path = os.path.join(output_folder, f'density_class_{column}.png')
        plt.savefig(density_path)
        plt.close()

"""
________________________________________________________
"""
"""
_______________________________________________________________________________________________

"""


@routes_bp.route('/bivariate-graphs-class/<string:dataset_id>', methods=['GET'])
def bivariate_graphs_class(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "Imputación"
        document = db[original_collection_name].find_one({"dataset_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        # Crear carpeta con nombre del identificador del dataset para almacenar gráficos
        output_folder = f"bivariate_graphs_class_{dataset_id}"
        os.makedirs(output_folder, exist_ok=True)

        # Generar y almacenar el gráfico pair plot
        pairplot_path = generate_pair_plot(original_dataset, output_folder)

        return jsonify({"pair_plot_url": pairplot_path})

    except Exception as e:
        return jsonify({"error": f"Error al generar el gráfico pair plot: {str(e)}"}), 500


def generate_pair_plot(dataset, output_folder):
    caca = request.json
    clases_objetivos = caca.get('caca')
    unique_classes = dataset[clases_objetivos].unique()
    # Filtrar el conjunto de datos para obtener solo las clases 2 y 4
    filtered_dataset = dataset[dataset[clases_objetivos].isin(unique_classes)]

    # Crear el gráfico pair plot
    pair_plot = sns.pairplot(filtered_dataset, hue=clases_objetivos, palette='husl')

    # Ajustes de la figura
    plt.title('Pair Plot por Clase')

    # Guardar el gráfico
    pairplot_path = os.path.join(output_folder, 'pair_plot.png')
    pair_plot.savefig(pairplot_path)
    plt.close()

    return pairplot_path



"""
_________________

"""


@routes_bp.route('/multivariate-graphs-class/<string:dataset_id>', methods=['GET'])
def multivariate_graphs_class(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "Imputación"
        document = db[original_collection_name].find_one({"dataset_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        # Crear carpeta con nombre del identificador del dataset para almacenar gráficos
        output_folder = f"multivariate_graphs_class_{dataset_id}"
        os.makedirs(output_folder, exist_ok=True)

        # Generar y almacenar el gráfico de correlación
        correlation_plot_path = generate_correlation_plot(original_dataset, output_folder)

        return jsonify({"correlation_plot_url": correlation_plot_path})

    except Exception as e:
        return jsonify({"error": f"Error al generar el gráfico de correlación: {str(e)}"}), 500

def generate_correlation_plot(dataset, output_folder):
    caca = request.json
    clases_objetivos = caca.get('caca')
    # Filtrar el conjunto de datos para obtener solo las clases 2 y 4
    filtered_dataset = dataset[dataset[clases_objetivos].isin([2, 4])]

    # Seleccionar solo las columnas numéricas
    numeric_columns = filtered_dataset.select_dtypes(include='number')

    # Calcular la matriz de correlación
    correlation_matrix = numeric_columns.corr()

    # Crear el gráfico de correlación usando Seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    # Ajustes de la figura
    plt.title('Matriz de Correlación de Columnas Numéricas')

    # Guardar el gráfico
    correlation_plot_path = os.path.join(output_folder, 'correlation_plot.png')
    plt.savefig(correlation_plot_path)
    plt.close()

    return correlation_plot_path


"""

_______________________
"""

"""

________________-

"""


@routes_bp.route('/pca/<string:dataset_id>', methods=['POST'])
def apply_pca(dataset_id):
    try:
        # Convertir dataset_id a ObjectId
        dataset_id_objectid = ObjectId(dataset_id)

        # Obtener el conjunto de datos según el dataset_id desde la colección original_datasets
        original_collection_name = "mi_coleccion"
        document = db[original_collection_name].find_one({"_id": dataset_id_objectid})

        if not document:
            return jsonify({"error": "Dataset no encontrado"}), 404

        # Extraer el array de información del documento
        dataset_information = document.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset_information)

        if original_dataset.empty:
            return jsonify({"error": "Conjunto de datos vacío"}), 400

        # Aplicar PCA
        transformed_dataset, pca_weights = apply_pca_to_dataset(original_dataset)

        # Crear una nueva versión del dataset con los datos transformados
        new_dataset_id = save_transformed_dataset_to_mongo(transformed_dataset)

        return jsonify({"pca_weights": pca_weights.tolist() if isinstance(pca_weights, np.ndarray) else pca_weights, "new_dataset_id": str(new_dataset_id)})

    except Exception as e:
        return jsonify({"error": f"Error al aplicar PCA: {str(e)}"}), 500


def apply_pca_to_dataset(dataset):
    # Seleccionar solo las columnas numéricas
    numeric_columns = dataset.select_dtypes(include='number')

    # Aplicar PCA
    pca = PCA()
    transformed_data = pca.fit_transform(numeric_columns)

    # Obtener los pesos de las componentes
    pca_weights = pd.DataFrame(pca.components_, columns=numeric_columns.columns)

    return transformed_data, pca_weights


def save_transformed_dataset_to_mongo(transformed_dataset):
    # Crear una nueva versión del dataset con los datos transformados
    transformed_collection_name = "transformed_datasets"
    result = db[transformed_collection_name].insert_one({"informacion": transformed_dataset.tolist() if isinstance(transformed_dataset, np.ndarray) else transformed_dataset})

    return result.inserted_id


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np

@routes_bp.route('/train/<string:dataset_id>/', methods=['POST'])
def train(dataset_id):
    try:
        # Obtener los datos del cuerpo de la petición (request)
        data = request.json

        # Obtener los parámetros requeridos
        algorithms = data.get("algorithms")
        option_train = data.get("option_train")
        normalization = data.get("normalization")
        clases_objetivos = data.get("caca")

        # Llamar a la función train_models con los parámetros requeridos
        result = train_models(dataset_id,algorithms, option_train, normalization,clases_objetivos)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {str(e)}"}), 500


def train_models(dataset_id, algorithms, option_train, normalization,clases_objetivos):
    try:
        # Obtener el dataset desde MongoDB
        collection_name = "Imputación"
        dataset_data = db[collection_name].find_one({"_id": ObjectId(dataset_id)})

        # Crear un DataFrame de pandas con el dataset
        dataset = dataset_data.get("informacion", [])

        # Crear un DataFrame de pandas con la información del array
        original_dataset = pd.DataFrame(dataset)
        # Obtener las features (X) y la variable objetivo (y)
        X = original_dataset.drop(columns=[clases_objetivos])
        y = original_dataset[clases_objetivos]

        # Normalizar los datos según la opción dada
        if normalization == 1:  # MinMax
            scaler = MinMaxScaler()
        elif normalization == 2:  # Standard Scaler
            scaler = StandardScaler()
        else:
            return {"error": "Opción de normalización no válida"}, 400

        X_normalized = scaler.fit_transform(X)

        # Inicializar modelos según los algoritmos dados
        models = []
        for algorithm in algorithms:
            if algorithm == 1:
                model = LogisticRegression()
            elif algorithm == 2:
                model = KNeighborsClassifier()
            elif algorithm == 3:
                model = SVC()
            elif algorithm == 4:
                model = GaussianNB()
            elif algorithm == 5:
                model = DecisionTreeClassifier()
            elif algorithm == 6:
                model = MLPClassifier(max_iter=1000)  # Ajustar según sea necesario
            else:
                return {"error": f"Algoritmo no válido: {algorithm}"}, 400

            pos_label = y.max()
            if option_train == 1:  # Hold out
                # Dividir los datos en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

                # Entrenar el modelo
                model.fit(X_train, y_train)

                # Calcular la precisión en el conjunto de prueba
                accuracy = model.score(X_test, y_test)

                # Entrenar el modelo
                model.fit(X_train, y_train)

                # Realizar predicciones en el conjunto de prueba
                y_pred = model.predict(X_test)

                # Calcular la matriz de confusión
                confusion = confusion_matrix(y_test, y_pred)

                # Calcular la precisión, recall y F1 Score
                precision = precision_score(y_test, y_pred,pos_label=pos_label)
                recall = recall_score(y_test, y_pred,pos_label=pos_label)
                f1 = f1_score(y_test, y_pred,pos_label=pos_label)

            elif option_train == 2:  # Cross Validation
                # Evaluar el modelo mediante validación cruzada
                scores = cross_val_score(model, X_normalized, y, cv=5)  # cv=5 significa 5 folds

                # Calcular la precisión promedio
                accuracy = scores.mean()

                # Obtener predicciones mediante validación cruzada
                y_pred = cross_val_predict(model, X_normalized, y, cv=5)  # cv=5 significa 5 folds

                # Calcular la matriz de confusión
                confusion = confusion_matrix(y, y_pred)

                # Calcular la precisión, recall y F1 Score
                precision = precision_score(y, y_pred,pos_label=pos_label)
                recall = recall_score(y, y_pred,pos_label=pos_label)
                f1 = f1_score(y, y_pred,pos_label=pos_label)


            else:
                return {"error": "Opción de entrenamiento no válida"}, 400

            # Guardar el modelo en MongoDB
            model_data = {
                "algorithm": algorithm,
                "accuracy": accuracy,
                "matriz_confusion":confusion.tolist() if isinstance(confusion, np.ndarray) else confusion,
                "precision":precision,
                "f1-Score":f1,
                "recall":recall,
                "dataset_id": ObjectId(dataset_id)
            }
            model_id = db["modelos"].insert_one(model_data).inserted_id

            models.append({
                "model_id": str(model_id),
                "algorithm": algorithm,
                "accuracy": accuracy,
                "matriz_confusion":confusion.tolist() if isinstance(confusion, np.ndarray) else confusion,
                "precision":precision,
                "f1-Score":f1,
                "recall":recall,
            })

        # Guardar la información del entrenamiento en MongoDB
        training_data = {
            "dataset_id": ObjectId(dataset_id),
            "models": models,
            "option_train": option_train,
            "normalization": normalization
        }
        training_id = db["entrenamientos"].insert_one(training_data).inserted_id

        return {
            "training_id": str(training_id),
            "models": models,
            "option_train": option_train,
            "normalization": normalization
        }

    except Exception as e:
        return {"error": f"Error al entrenar modelos: {str(e)}"}, 500


"""

________________________
"""





@routes_bp.route('/results/<string:train_id>', methods=['GET'])
def get_results(train_id):
    try:
        # Obtener la información del entrenamiento desde MongoDB
        training_data = db["entrenamientos"].find_one({"_id": ObjectId(train_id)})

        if training_data is None:
            return jsonify({"error": "Entrenamiento no encontrado"}), 404

        # Obtener las métricas de cada modelo
        models_metrics = []
        for model_info in training_data.get("models", []):
            model_metrics = {
                "model_id": model_info.get("model_id"),
                "algorithm": model_info.get("algorithm"),
                "accuracy": model_info.get("accuracy"),
                "precision": model_info.get("precision"),
                "recall": model_info.get("recall"),
                "f1_score": model_info.get("f1-Score"),
                "confusion_matrix": model_info.get("matriz_confusion")
            }
            models_metrics.append(model_metrics)

        # Devolver la información de métricas de cada modelo
        return jsonify({"models_metrics": models_metrics})

    except Exception as e:
        return jsonify({"error": f"Error al obtener resultados: {str(e)}"}), 500

"""

_________________________________________

"""


@routes_bp.route('/prediction/<string:train_id>', methods=['GET'])
def make_prediction(train_id):
    try:
        # Obtener la información del entrenamiento desde MongoDB
        training_data = db["entrenamientos"].find_one({"_id": ObjectId(train_id)})

        if training_data is None:
            return jsonify({"error": "Entrenamiento no encontrado"}), 404

        # Obtener los modelos del entrenamiento
        models = training_data.get("models", [])

        if not models:
            return jsonify({"error": "No hay modelos entrenados"}), 400

        # Encontrar el modelo con la mejor métrica F1 Score
        best_model = max(models, key=lambda x: x.get("f1-Score", 0.0))

        #ATASCAMIENTOOOOOOOOOOO

        # Obtener el dataset de prueba desde el body de la solicitud
        test_data = request.json.get("test_data")

        if test_data is None:
            return jsonify({"error": "Datos de prueba no proporcionados"}), 400

        # Realizar la predicción con el mejor modelo
        model_algorithm = best_model.get("algorithm")
        model = get_model_instance(model_algorithm)

        # Entrenar el modelo con el conjunto de entrenamiento completo
        X_train = np.array(training_data.get("X_train"))
        y_train = np.array(training_data.get("y_train"))
        model.fit(X_train, y_train)

        # Realizar la predicción con el conjunto de prueba
        predictions = model.predict(test_data)

        # Devolver las predicciones como respuesta
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": f"Error al realizar la predicción: {str(e)}"}), 500


def get_model_instance(algorithm):
    # Devolver una instancia del modelo basado en el algoritmo
    if algorithm == 1:
        return LogisticRegression()
    elif algorithm == 2:
        return KNeighborsClassifier()
    elif algorithm == 3:
        return SVC()
    elif algorithm == 4:
        return GaussianNB()
    elif algorithm == 5:
        return DecisionTreeClassifier()
    elif algorithm == 6:
        return MLPClassifier(max_iter=1000)
    else:
        return None



"""

______________________________
"""
@routes_bp.route('/delete_all', methods=['DELETE'])
def delete_all_documents():
    try:
        # Eliminar todos los documentos de la colección
        result = db.mi_coleccion.delete_many({})

        # Verificar el resultado de la operación
        if result.deleted_count > 0:
            return jsonify({"success": "Todos los documentos eliminados correctamente"}), 200
        else:
            return jsonify({"warning": "No se encontraron documentos para eliminar"}), 404

    except Exception as e:
        return jsonify({"error": f"Error al eliminar documentos: {str(e)}"}), 500
