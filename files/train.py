from sklearn.linear_model import LogisticRegression
import argparse
from sklearn.metrics import accuracy_score
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


# +
# def replace_categorical_values(df):
#    sex = {0: 'Female', 1: 'Male'}
#    cp = {0: "typical_angina", 
#          1: "atypical_angina", 
#          2:"non-anginal pain",
#          3: "asymtomatic"}
#    exang = {0: "No", 1: "Yes"}
#    fbs = {0: "less_than_120", 1: "greater_than_120"}
#    slope = {0: "upsloping", 1: "flat", 2: "downsloping"}
#    thal = {1: "fixed_defect", 2: "reversible_defect", 3:"normal"}
#    restecg = {0: 'normal', 1: 'ST-T_wave_abnormality' , 2: 'left_ventricular_hypertrophy'}

#   df.sex = df.sex.replace(sex)
#    df.cp = df.cp.replace(cp)
#    df.exang = df.exang.replace(exang)
#    df.fbs = df.fbs.replace(fbs)
#    df.slope = df.slope.replace(slope)
#    df.thal = df.thal.replace(thal)
#    df.restecg = df.restecg.replace(restecg)
    
#    return df
# -

def transform_data(df):
    # ca extra cateogry 4, remove that and fill with median
    df.loc[df.ca == 4, 'ca'] = np.NaN
    df.ca = df.ca.fillna(df.ca.median())
    
    # thal has extra category 0, remove that and fill with median
    df.loc[df.thal == 0, 'thal'] = np.NaN
    df.thal = df.thal.fillna(df.thal.median())
    
    # There is one duplicate entry, remove that
    df.drop_duplicates(keep='first',inplace=True)
        
    # Changing name of columns to understand the data better 
    df.columns = df.columns = ['age', 'sex', 'chest_pain_type', 'resting_BP', 'cholesterol', 'fasting_blood_sugar', 'rest_ECG', 'max_heart_rate',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

    categorical_values = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ECG', 
                          'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalassemia']
    
    # encode the data
    data = pd.get_dummies(df, columns=categorical_values, drop_first=True)
    
    return data


def clean_data(data):
    df = data.to_pandas_dataframe()
    
    # changing the categorical data
    data = transform_data(df)
    
    # print(data.columns)
    print("Shape of dataset before split:", data.shape)

    # Dropping the target column 'target' from the dataset
    y_df = data.target
    x_df = data.drop('target', axis = 1)
    
    # The data has no NaN/ null values. 
    
    return x_df, y_df

# +
# url_path = "https://raw.githubusercontent.com/bharati-21/AZMLND-Capstone-Project/master/files/heart.csv"
# ds = Dataset.Tabular.from_delimited_files(path=url_path)

# X, y = clean_data(ds)
# print("Data Shape after split:", X.shape, y.shape)

# X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.2, random_state = 42)

# log_reg = LogisticRegression()

# model = log_reg.fit(X_train,y_train)
# prediction = model.predict(X_test)
# accuracy_score(y_test,prediction)
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
   
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model,filename='outputs/model.pkl')

if __name__ == '__main__':
    main()


