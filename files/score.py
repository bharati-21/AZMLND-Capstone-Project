
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
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hyperdrive-heart-disease.pkl')
    model = joblib.load(model_path)

    
def run(data):
    try: 
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
    
