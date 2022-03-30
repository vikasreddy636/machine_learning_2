import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")


x = pd.read_csv(r'LNIP_super_all.csv')

training_set,test_set = train_test_split(x,test_size=0.2,random_state=0)
X_train = training_set.iloc[:,0:59].values
Y_train = training_set.iloc[:,59].values
X_test = test_set.iloc[:,0:59].values
Y_test = test_set.iloc[:,59].values

#################MLP Classifier#################
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, Y_test)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("Accuracy of MLP classifier: ",accuracy*100)

#################Optimized SVM with RBF Kernel###############
#For getting 98% accuracy feed C=1100 and Gamma = 1000

classifier = SVC(C=1100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1000, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
classifier.fit(X_train,Y_train)
SVM_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test,SVM_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Optimized SVM with RBF kernel "+str(accuracy*100))#+" by optimized RBF with 'C'= "+str(1100)+" Gamma value= "+str(1000))

########################Linear SVM Classifier###############
SVM = svm.LinearSVC()
SVM.fit(X_train,Y_train)
SVM_pred = SVM.predict(X_test)
cm = confusion_matrix(Y_test,SVM_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Linear SVM For The Given Dataset: ", accuracy*100)

##############################Naive Bayes################
gnb = GaussianNB() 
gnb.fit(X_train, Y_train) 
y_pred = gnb.predict(X_test) 
cm = confusion_matrix(Y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
#print("\nAccuracy of Naive Bayes Classsifier:", accuracy*100)

######################## Random Forest Classifier#################
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
rfc_predict = rfc.predict(X_test)
cm = confusion_matrix(Y_test,rfc_predict)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Random Forest For The Given Dataset :", accuracy*100)

###################### XG Boost############################
model = XGBClassifier()
model.fit(X_train, Y_train)
#make predictions for test data
y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
predictions = [round(value) for value in y_pred]
#evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("\nAccuracy of XGBoost: %.2f%%" % (accuracy * 100.0))

###################### Naive Bayes ########################
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
predictions = [round(value) for value in y_pred]
#evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("\nAccuracy of Naive Bayes: %.2f%%" % (accuracy * 100.0))