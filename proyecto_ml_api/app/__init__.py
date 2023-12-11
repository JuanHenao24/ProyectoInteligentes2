from flask import Flask
from flask_pymongo import PyMongo

mongo = PyMongo()

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('config.py')

    mongo.init_app(app)

    from proyecto_ml_api.app.routes import main
    app.register_blueprint(main)

    return app