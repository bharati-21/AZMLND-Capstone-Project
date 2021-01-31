from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    # Changing name of columns to understand the data better 
    data.columns = ['age', 'sex', 'chest_pain_type', 'resting_BP', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    
    # Dropping the target column 'target' from the dataset
    y_df = data.target
    x_df = data.drop('target', axis = 1)
    
    # The data has no NaN/ null values. 
    # Encode the categorical column 'thalassemia' using LabelEncoder
    label_encoder = LabelEncoder()
    x_df["thalassemia"] = label_encoder.fit_transform(x_df['thalassemia'])
    
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # Create TabularDataset using TabularDatasetFactory
    # Data is located at: https://raw.githubusercontent.com/bharati-21/AZMLND-Capstone-Project/master/files/heart.csv
    
    url_path = "https://raw.githubusercontent.com/bharati-21/AZMLND-Capstone-Project/master/files/heart.csv"
    ds = Dataset.Tabular.from_delimited_files(path=url_path)

    # clean the data
    x, y = clean_data(ds)

    #  Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
   
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model,filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()


