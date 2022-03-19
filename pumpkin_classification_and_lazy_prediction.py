# -*- coding: utf-8 -*-
"""Pumpkin classification and  Lazy Prediction.ipynb
"""
import os
import pickle

import openpyxl as openpyxl
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier

train = pd.read_excel('Pumpkin_Seeds_Dataset.xlsx', engine='openpyxl')
# train =
train.head()

train.drop_duplicates(keep=False, inplace=False)

train.nunique().sort_values(ascending=False)

round(train.isnull().sum() * 100 / len(train), 2).sort_values(ascending=False)

round(train['Class'].value_counts() * 100 / len(train), 2)

train['Class'] = train['Class'].replace({'Çerçevelik': 0, 'Ürgüp Sivrisi': 1})
train['Class'].value_counts()

y = train.pop('Class')
X = train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

clf = LazyClassifier(verbose=0, predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

predictions.head()

for i in predictions.columns.tolist():
    print('\t\t', i, '\n')
    print(classification_report(y_test, predictions[i]), '\n')

svc = SVC(kernel='linear')

print("Fitting Model..... Linear SVC")
svc.fit(X_train, y_train)
print("Model Fitted")

svc.score(X_test, y_test)

# params = {'C': [0.1, 1], 'gamma': [0.001, 0.01, 0.1]}
# searcher = GridSearchCV(svc, params)
# searcher.fit(X_train, y_train)

#
# with open('final_model_pumpkin_classification-svc', 'wb') as files:
#     pickle.dump(svc, files)
#
# with open('final_model_pumpkin_classification-searcher', 'wb') as files:
#     pickle.dump(svc, files)

pickle.dump(svc, open("svc_model.pkl", "wb"))

# pickle.dump(searcher, open("searcher_model.pkl", "wb"))
