# MACHINE LEARNING ENGINEER WITH MICROSOFT AZURE NANODEGREE PROGRAM BY UDACITY - CAPSTONE PROJECT  

## TABLE OF CONTENTS
* [Project Overview](#project-overview)
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
* [Automated ML](#automated-ml)
  * [Results](#results)
* [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Results](#results)
* [Model Deployment](#model-deployment)
* [Screen Recording](#screen-recording)
<hr/>

## Project Overview
> This project is part of the Udacity's Machine Learning Engineer with Microsoft Azure Nanodegree Project.

* In this project, two models were created:
  * Using Automated ML (AutoML)
  * Using customized model whose hyperparameters were tuned using ***HyperDrive***.
* The performance of both the models were compared and the best _performing model was deployed_.

## Project Set Up and Installation
* In this project, the Azure ML lab offered by Udacity was used. Hence, the Workspace was already set up and ready.
* A compute instance `compute-project` was created with STANDARD_DS3_V2 VM size. 
![Image of Compute Instance](Images/compute_instance.png)
* The starter files from this [project repository](https://github.com/udacity/nd00333-capstone) were forked and cloned to the workspace.
<hr/>

## Dataset
* A dataset external to the Azure ML ecosystem was chosen to train the AutoML and HyperDrive runs.
### Overview
* To train the models, the [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) dataset was used from [Kaggle](https://www.kaggle.com).
* This dataset originally contains 76 attributes and was taken from the [UCI Machine Learning Repository Archive](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
* But all published experiments refer to using a subset of 14 of them. The following features are part of the dataset:
  * `Age (age)`: The person’s age in years  
  * `Sex (sex)`: The person's sex 
    * 1: male
    * 0: female
  * `Chest pain type (cp)`: The type of chest pain the person experiences
    * 0: Typical Angina Chest Pain
    * 1: Atypical Angina Chest Pain
    * 2: Non-Anginal Chest Pain
    * 3: Asymptomatic Chest Pain
  * `Resting Blood Pressure (trestbps)`: Resting Blood Pressure (in mm Hg) of the person on admission to the hospital
  * `Cholesterol (chol)`: Serum cholestoral in mg/dl
  * `Fasting Blood Sugar (fbs)`: If the person's Fasting Blood sugar level is greater or less than 120 mg/dl
    * 0: False (less than 120 mg/dl)
    * 1: True (greater than 120 mg/dl)
  * `Resting Electrocardiographic (restecg)` : The person's ECG result when at rest to find whether T wave exists.
    * 0: Normal
    * 1: ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * 2: Probable or Definite Ventricular Hypertrophy
  * `Maximum Heart Rate Achieved (thalach)`: The maximum heart rate (heart beat per minute) of the person
  * `Exercise induced Angina (exang)`: If the person had an Anginal chest pain due to exercise. (Angina is caused when there is limited blood supply to the heart)
    * 0: No
    * 1: Yes
  * `ST depression induced by exercise (oldpeak)`: Pressure of the ST segment of the wave in person's ECG compared to rest ECG
  * `Slope (slope)`: Slope of the peak exercise ST segment of the wave in ECG of the person. 
    * 0: upsloping
    * 1: flat
    * 2: downslopin
  * `Number of major vessels (ca)`: Number of major vessels colored by flourosopy procedure of the person
    * 0, 1, 2 and 3
  * `Thalassemia (thal)`: Categories of complications of the thalassemia defect. 
    * 1: Normal
    * 2: Fixed Defect
    * 3: Reversable Defect
  * `Target (target)`: The presence of absence of heart disease in the patient.
    * 0: Absence
    * 1: Presence
### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
* The goal of the project was to train the model to predict whether a patient has heart disease or not.
* The features used to train the model were:
  1. `Age (age)`
  1. `Sex (sex)`
  1. `Chest pain type (cp)`
  1. `Resting Blood Pressure (trestbps)`
  1. `Cholesterol (chol)`
  1. `Fasting Bloos Sugar (fbs)`
  1. `Resting Electrocardiographic (restecg)`
  1. `Maximum Heart Rate Achieved (thalach)`
  1. `Exercise induced Angina (exang)`
  1. `ST depression induced by exercise (oldpeak)`
  1. `Slope (slope)`
  1. `Number of major vessels (ca)`
  1. `Thalassemia (thal)`
* The prediction column is `Target (target)` which is used by the model to predict whether a person has heart disease or not
### Access
*TODO*: Explain how you are accessing the data in your workspace.
* The dataset was uploaded and registered to the Azure default datastore as a Tabular Dataset using the [csv file](heart.csv).
<hr/>

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
