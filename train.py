print('running the train.py file ..............................................')
print('-'*1000)
print()



from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace, Dataset
print('loaded azureml dependencies!')

print('importing libraries ...')
# Import libraries
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import joblib
print('libraries imported!')

ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws=Workspace.from_config(auth=ia)

# subscription_id = ''
# resource_group = ''
# workspace_name = ''

# ws = Workspace(subscription_id, resource_group, workspace_name)


# dataset = Dataset.get_by_name(ws, name='dataset_name')
# data_folder='data'
# dataset.download(target_path=data_folder, overwrite=True)


# def registeModel(model_path, model_name):
# 	azm.register(workspace=ws, model_path=model_path, model_name=model_name)

# parser=argparse.ArgumentParser()
# parser.add_argument('--data_path',type=str, help='Paht where the images are stored')
# parser.add_argument('--output_dir', type=str, help='output directory')
# input_path = args.data_path

# code for training and initialising all the parmas

# print('*'*100)
# print('Everything is running fine ')
# print('*'*100)

# run = Run.get_context()
# run.tag("Description", "trained the model")
# run.log('koi metric','konsi metric re baba')
# run.log('sensitivity', 'acha')

# run.complete()



print(ws)

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg

# load the diabetes dataset
print("Loading Data...")
# load the diabetes dataset
diabetes = pd.read_csv('data/diabetes.csv')

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

# # Register the model
# run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
#                 tags={'Training context':'Script'},
#                 properties={'AUC': str(np.float(auc)), 'Accuracy': str(auc)})


run.complete()