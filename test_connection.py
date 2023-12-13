from pymongo import MongoClient
import json

# Cargar la configuración desde el archivo
with open('config.json') as config_file:
    config = json.load(config_file)

# Construir la cadena de conexión
mongo_uri = f"mongodb+srv://{config['mongo']['username']}:{config['mongo']['password']}@{config['mongo']['host']}/{config['mongo']['database']}"

# Intentar establecer la conexión y realizar una operación de prueba
try:
    client = MongoClient(mongo_uri)
    db = client[config['mongo']['database']]

    # Operación de prueba: imprimir el resultado de una consulta
    result = db.test_collection.find_one()
    print("Conexión exitosa. Resultado de la operación de prueba:")
    print(result)

except Exception as e:
    print(f"Error al conectar a MongoDB: {str(e)}")