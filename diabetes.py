# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:09:38 2021

@author: asabi
"""

#standard libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree

url = url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/diabetes.csv'

data = pd.read_csv(url)
data.head()
data.columns
data.groupby('Outcome').aggregate({'BMI':np.mean,'Glucose':np.mean,'Age':np.mean,'BloodPressure':np.mean})
data.Outcome.value_counts()
data.shape
X=data.drop('Outcome',axis=1) #features
y=data['Outcome'] #target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=1)
X_train.shape
X_test.shape
X_train.head()
y_train.shape 
y_test.shape 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf
help(clf)
#Train the classifer
clf=clf.fit(X_train,y_train)
y_train
#apply model on the test set
y_pred=clf.predict(X_test)
y_pred
y_test
#compare for accuracy
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
(116+46)/len(y_test) #accuracy
metrics.accuracy_score(y_test, y_pred)


#classification report --check
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

#prune tree
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html
text_representation2 = tree.export_text(dtree)
print(text_representation2)
data.columns
text_representation3 = tree.export_text(dtree, feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'],  decimals=0, show_weights=True, max_depth=3)  #keep changing depth values
print(text_representation3)

