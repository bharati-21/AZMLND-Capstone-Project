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


def transform_data(data):
    
    # categorical columns in the dataset
    categorical_columns = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalassemia']

    # converting the categorical columns to object type
    data['sex'] = data['sex'].astype('object')
    data['chest_pain_type'] = data['chest_pain_type'].astype('object')
    data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')
    data['rest_ecg'] = data['rest_ecg'].astype('object')
    data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')
    data['st_slope'] = data['st_slope'].astype('object')
    data['num_major_vessels'] = data['num_major_vessels'].astype('object')
    data['thalassemia'] = data['thalassemia'].astype('object')
    
    # encode the data
    dataset = pd.get_dummies(data, columns= categorical_columns, drop_first= True)
    
    return dataset


def clean_data(data):
    # Changing name of columns to understand the data better 
    data_df = data.to_pandas_dataframe()
    data_df.columns = ['age', 'sex', 'chest_pain_type', 'resting_BP', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    
    # changing the categorical data
    dataset = transform_data(data_df)
    
    print(dataset.columns)
    print(dataset.shape)

    # Dropping the target column 'target' from the dataset
    y_df = dataset.target
    x_df = dataset.drop('target', axis = 1)
    
    # The data has no NaN/ null values. 
    
    return x_df, y_df

# +
url_path = "https://raw.githubusercontent.com/bharati-21/AZMLND-Capstone-Project/master/files/heart.csv"
ds = Dataset.Tabular.from_delimited_files(path=url_path)

x, y = clean_data(ds)
print(x.shape, y.shape)


# -

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


