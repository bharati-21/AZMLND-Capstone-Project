import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://470f5572-e98b-468b-8a0e-d0e249caed1a.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'NtorvXegdH9EAhMDhbhlhVzmH0Z9XskK'

# +
# Two sets of data to score, so we get two results back

data = {
    "data": 
        [
            {
                "age": 67.0,
                "resting_BP": 120.0,
                "cholesterol": 229.0,
                "max_heart_rate": 129.0,
                "st_depression": 2.6,
                "sex_1": 1.0,
                "chest_pain_type_1": 0.0,
                "chest_pain_type_2":  0.0,
                "chest_pain_type_3":  0.0, 
                "fasting_blood_sugar_1": 0.0, 
                "rest_ECG_1": 0.0, 
                "rest_ECG_2": 0.0, 
                "exercise_induced_angina_1": 1.0,
                "st_slope_1": 1.0,
                "st_slope_2": 0.0, 
                "num_major_vessels_1.0": 0.0, 
                "num_major_vessels_2.0": 1.0, 
                "num_major_vessels_3.0": 0.0,
                "thalassemia_2.0":  0.0,
                "thalassemia_3.0":  1.0
            },
            {
                "age": 64.0,
                "resting_BP": 130.0,
                "cholesterol": 303.0,
                "max_heart_rate": 122.0,
                "st_depression": 2.0,
                "sex_1": 0.0,
                "chest_pain_type_1": 0.0,
                "chest_pain_type_2":  0.0,
                "chest_pain_type_3":  0.0, 
                "fasting_blood_sugar_1": 0.0, 
                "rest_ECG_1": 1.0, 
                "rest_ECG_2": 0.0, 
                "exercise_induced_angina_1": 0.0,
                "st_slope_1": 1.0,
                "st_slope_2": 0.0, 
                "num_major_vessels_1.0": 0.0, 
                "num_major_vessels_2.0": 1.0, 
                "num_major_vessels_3.0": 0.0,
                "thalassemia_2.0":  1.0,
                "thalassemia_3.0":  0.0
            }
        ]
    }


# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)
# -

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


