from flask import Flask
from proyecto_ml_api.app.routes import api_bp



app = Flask(__name__)

# Registrar las rutas de la API
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True)