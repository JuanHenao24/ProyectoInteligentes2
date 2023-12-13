from pymongo import MongoClient
import json

def get_mongo_uri(config):
    username = config['mongo']['username']
    password = config['mongo']['password']
    host = config['mongo']['host']
    database = config['mongo']['database']

    return f"mongodb+srv://{username}:{password}@{host}/{database}?retryWrites=true&w=majority"

def connect_to_mongo(config):
    mongo_uri = get_mongo_uri(config)
    client = MongoClient(mongo_uri)
    return client[config['mongo']['database']]