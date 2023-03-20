import json


def load_json(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data
