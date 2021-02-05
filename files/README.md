# MACHINE LEARNING ENGINEER WITH MICROSOFT AZURE NANODEGREE PROGRAM BY UDACITY - CAPSTONE PROJECT  

## TABLE OF CONTENTS
* [Project Overview](#project-overview)
  * [Projct Architecture](@project-architecture)
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
### Project Architecture
* ![Image of Projct Architecture](Images/CP_Architecture.png)

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
  * `Age (age)`, `Sex (sex)`, `Chest pain type (cp)`, `Resting Blood Pressure (trestbps)`, `Cholesterol (chol)`, `Fasting Bloos Sugar (fbs)`, `Resting Electrocardiographic (restecg)`, `Maximum Heart Rate Achieved (thalach)`, `Exercise induced Angina (exang)`, `ST depression induced by exercise (oldpeak)`, `Slope (slope)`, `Number of major vessels (ca)`, `Thalassemia (thal)`
* The prediction column is `Target (target)` which is used by the model to predict whether a person has heart disease or not
### Access
*TODO*: Explain how you are accessing the data in your workspace.
* The dataset was uploaded and registered to the Azure default datastore as a Tabular Dataset using the [csv file](heart.csv).
<hr/>

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
* ***Automated Machine Learning (AutoML)***
  * AutoML is the process of automating the time consuming, iterative tasks of machine learning model development. 
  * It helps in developeing ML models with high scale, efficiency, and productivity all while sustaining model quality. 
* ***Steps involved in developing the model:***
  * Training a model with the given dataset was identified to be a `_classification_` task since the goal was to predict whether a person has heart disease or not.
  * In this project `_Python SDK_` was used to complete the task. But the Azure ML studio designer can also be used to train the model.
  * A remote ML compute cluster, `_cpu-cluster_`, was used to train the models on the dataset. The dedicated virtual machine size of the compute cluster was `STANDARD_DS12_V2`, with `1 minumum and 6 maximum number of nodes` and `CPU as the Processing unit`.
  * AutoMLConfig Class was used to create the configurations for submitting the AutoML run experiment.   
    * The settings created for the AutoML run was:
      * `Experiment Timeout (experiment_timeout_minutes)`: Maximum amount of time (in minutes) that all iterations combined can take before the experiment terminates.
      * `Primary Metric (primary_metric)`: The primary metric which is used to evaluate every run. In this case, accuracy is the primary metric to be evaluated.
      * `Cross Validations (n_cross_validations)`: Specifies the number of cross validations that needs to be performed on each model by splitting the dataset into n subsets.
      ``` 
      automl_settings = {
       "experiment_timeout_minutes": 30,
       "primary_metric": 'accuracy',
       n_cross_validations = 5,
      }
      ```
      
    * The AutoMLConfg object that would be submitted to the experiment was defined as followed:
      * Task to be performed (task): The tpye of task that needs to be run such as classification, regression, forecasting etc. In this project classification is the task to be performed.
      * Training Data (training_data) = The TabularDataset that contains the training data.
      * Label Column (label_column_name): Name of the column that needs to be predicted. In this case the column that contains "yes" or "no" to perform classification.
      * Compute Target (compute_target): The cluster used to run the experiment on.
      ```
      automl_config = AutoMLConfig (
        task = 'classification',
        training_data = train_data,
        label_column_name = "target",
        enable_onnx_compatible_models = True,
        compute_target = cpu_cluster,
        **automl_settings
      )
      ```
  * Submit the training run.
    * The run was submitted to the experiment and AutoMLConfig object was passed as a parameter to the run"
    ```
    remote_run = experiment.submit(automl_config, show_output = True)
    ```
### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
* Once submitted the progress of the run was observed via the run widget of the _`RunDetails`_ class in the Jupyter notebook. 
  ```
  from azureml.widgets import RunDetails
  RunDetails(remote_run).show()
  ```
  ![Image of Run Widget](Images/autoML_run_widget.png)
* The following Algorithms were used on the dataset to retrieve the trained model:
  1. `LogisticRegression`
  1. `XGBoostClassifier`
  1. `LightGBM`
  1. `RandomForest`
  1. `SVM`
  1. `GradientBoosting`
  1. `ExtremeRandomTree`
  1. `KNN`
  1. `VotingEnsemble`
* The best model obtained post training with the highest accuracy was a `VotingEnsemble` algorithm with an accuracy of _`0.8380`_. 
* The best model was then registered with the provided workspace using the _`register()`_ method of _`Model`_ class.
  ```
  description = "AutoML model trained on the Kaggle Heart Disease UCI Dataset"
  joblib.dump(fitted_model, filename="outputs/automl-heart-disease.pkl") # saving the model locally
  automl_model = remote_run.register_model(model_name='automl-heart-disease', description=description)
  ```
  * `workspace`: Workspace name to register the model with.
  * `model_name`: The name to register the model with.
  * `description`: A text description of the model.

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
