import pickle
import json
import numpy
from azureml.core.model import Model
import joblib


def init():
    global model

    # load the model from file into a global object
    model_path = Model.get_model_path(model_name="diabetes_model.pkl")
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result = model.predict(data)
        return json.dumps({"diabetes result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"An error occured while prediction in Score.py file......": result})