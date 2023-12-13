from flask import Flask
from dbconection import connect_to_mongo
import json
from proyecto_ml_api.app.routes import routes_bp

app = Flask(__name__)
# Registrar el Blueprint
app.register_blueprint(routes_bp)


# Cargar la configuración desde el archivo
with open('config.json') as config_file:
    config = json.load(config_file)

# Establecer la conexión a MongoDB
db = connect_to_mongo(config)

@app.route('/')
def index():
    return 'Hello, MongoDB!'

if __name__ == '__main__':
    app.run(debug=True)