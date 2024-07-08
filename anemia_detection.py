import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

FOREST_NO = 100
CSV_PATH = '/home/robosobo/ML_code/Datasets/anemia_data/anemia.csv'
MODEL_FILENAME = 'anemia_predictor.joblib'
SCALER_FILENAME = 'scaler.joblib'

data_frame = pd.read_csv(CSV_PATH)

print('First 5 data entries: ')
print(data_frame.head())

X = data_frame.drop(['Number', 'Anaemic'], axis=1)
Y = data_frame['Anaemic']

X['Sex'] = X['Sex'].map({'M': 0, 'F': 1})

# To handle missing values (NaN - Not a Number) in the dataset
# 'most_frequent': it replaces NaN with the most frequent value in the column
imputer = SimpleImputer(strategy='most_frequent')
X['Sex'] = imputer.fit_transform(X[['Sex']])

X = X.astype(float)

print('\nFeature Data:')
print(X)
print('\nTarget Data:')
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=FOREST_NO ,random_state=42)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f'\nAccuracy of our model: {accuracy:.4f}')

print('\nReport of our model: ')
report = classification_report(Y_test, Y_pred)
print(report)

dump(classifier, MODEL_FILENAME)
dump(scaler, SCALER_FILENAME)
print(f'\n|| MODEL IS SAVED ||')
