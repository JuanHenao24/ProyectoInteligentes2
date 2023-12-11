from flask import Blueprint, jsonify, request

api_bp = Blueprint('api', __name__)
main = Blueprint('main', __name__)

@api_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Verifica el encabezado 'Content-Type'
        if request.headers['Content-Type'] != 'application/json':
            return jsonify({'error': 'Unsupported Media Type'}), 415  # 415 - Unsupported Media Type

        # Obtén datos del cuerpo de la solicitud
        data = request.get_json()

        # Realiza la predicción usando tu modelo ML (puedes agregar esta parte según tu lógica)
        result = {'prediction': 'Puedes agregar el resultado de tu modelo aquí'}

        # Devuelve el resultado como JSON
        return jsonify(result)
    else:
        # Maneja solicitudes GET (puedes personalizar este bloque según tus necesidades)
        return jsonify({'message': 'Esta es una respuesta para solicitudes GET en /predict'})

from flask import Blueprint, request, jsonify
from flask_pymongo import PyMongo
import pandas as pd
import json


@api_bp.route('/load', methods=['POST'])
def load_excel():
    try:
        file = request.files['file']

        # Verificar si el archivo tiene un nombre válido
        if file.filename == '':
            return jsonify({"error": "Nombre de archivo no válido"}), 400

        # Verificar si el archivo tiene una extensión válida (puedes ajustar según tus necesidades)
        allowed_extensions = {'xlsx', 'xls'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Extensión de archivo no válida"}), 400

        # Leer el archivo Excel
        df = pd.read_excel(file)

        # Convertir a formato JSON
        json_data = df.to_json(orient='records')

        # Almacenar en MongoDB
        PyMongo.db.documents.insert_many(json.loads(json_data))

        return jsonify({"message": "Documentos cargados exitosamente"})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Ocurrió un error en el servidor"}), 500