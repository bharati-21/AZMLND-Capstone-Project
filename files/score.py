
import json
import numpy as np
import os
import pickle
# import sklearn.external.joblib as extjoblib
import joblib
# from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path('./outputs/model.joblib')
    model = joblib.load(model_path)

def run(data):
    data = np.array(json.loads(raw_data)['data'])
    result = model.predict(data)
    return json.dumps(result.tolist())
    
