# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:00:44 2021

@author: Rony's PC
"""

""" 
Here I take an economic data tabular clients bank to be inqueried as classification
by some types SVM models.
"""

"""
0. Introduction 
I try to compare both simple and kernel SVMs. I studied the intuition behind the
SVM algorithm and how it can be implemented with Python's Scikit-Learn library. 
I also studied different types of kernels that can be used to implement 
kernel SVM.
All that were done via small data of customers bank as classification.
I try to prevent any Dimensional Data Curse - no onehot. 
I take some comparison on model's SVM'
"""

"""
1. Libraries to import.
""" 

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

"""
2. Data to be louded.
"""

df = pd.read_csv(r"C:/Users/Rony's PC/OneDrive/ML/exercise/ML/Unsuprvised/summerizing clustering/bankCustomers.csv")
# First droping all features unnecessary. 
df = df.drop(['CLIENTNUM', 'Customer_Age',
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
"""
3. Inquery data.
"""

# To prevent any Dimensional Data Curse I don't work with onehot to duplicate Columns 

def Attrition_Flag2num(x):
    if x == 'Attrited Customer':
        return 1
    if x == "Existing Customer":
        return 0
df['Attrition_Flag'] = df['Attrition_Flag'].apply(Attrition_Flag2num)


        
# Secondly I begin to transform every feature needed to number.
def Gender2num(x):
    if x == 'M':
        return 1
    if x == "F":
        return 0
df['Gender'] = df['Gender'].apply(Gender2num)
                              

def Education_Level2num(x):
    if x == 'Uneducated':
        return 0
    if x == 'Unknown':
        return 0
    if x == 'College':
        return 1
    if x == 'Post-Graduate':
        return 1
    if x == 'Graduate':
        return 2
    if x == 'High School':
        return 3
    if x == 'Doctorate':
        return 4
df['Education_Level'] = df['Education_Level'].apply(Education_Level2num)
                   

def Marital_Status2num(x):
    if x == 'Unknown':
        return 0
    if x == 'Single':
        return 1
    if x == 'Divorced':
        return 1
    if x == 'Married':
        return 2
df['Marital_Status'] = df['Marital_Status'].apply(Marital_Status2num)

def Card_Category2num(x):
    if x == 'Blue':
        return 0
    if x == 'Silver':
        return 1
    if x == 'Gold':
        return 2
    if x == 'Platinum':
        return 3
df['Card_Category'] = df['Card_Category'].apply(Card_Category2num)
                      

# Inwuery of 'Income_Category' category feature :    
print(df['Income_Category'])
# To prevent Linkage I work with small nubers.
def Income_Category2income_int(x):
    if x == '$120K +':
        return 13
    if x == '$80K - $120K':
        return 10
    if x == '$60K - $80K':
        return 7
    if x == '$40K - $60K':
        return 5
    if x == 'Less than $40K':
        return 2
    if x == 'Unknown':
        return 0
df['Income_Category'] = df['Income_Category'].apply(Income_Category2income_int)


df.dtypes
df.info()



X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']       


"""
Implementing SVM with Scikit-Learn
-----------------------------------
-----------------------------------
"""

"""
4. Dividing data into training and test sets.  
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

"""
5. Fitting method of SVC class.
"""

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

"""
6. Making Predictions.
"""

y_pred = svclassifier.predict(X_test)

"""
7. Evaluating the Algorithm.
"""

print('Simple SVM')
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



"""
Implementing Kernel SVM with Scikit-Learn.
------------------------------------------
------------------------------------------
explication en francais:
    En apprentissage automatique, le noyau polynomial est une fonction
    noyau couramment utilisée avec les machines à vecteurs de support (SVMs)
    et d'autres modèles à noyaux. Il représente la similarité des vecteurs
    (échantillons d'apprentissage) dans un espace de degré polynomial plus
    grand que celui des variables d'origine, ce qui permet un apprentissage
    de modèles non-linéaires. 
"""

"""
1. - 4. Allthe steps are the same.
"""

X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)       



"""
All the first steps are the same as simple SVM, so
I jump directly to Polynomial Kernel.
"""

"""
5. Fitting method of SVC class with Polynomial Kernel.
"""

"""
Polynomial Kernel
The characteristic (implicit) space of a polynomial kernel is equivalent to
that of polynomial regression, but without the combinatorial explosion
of the number of parameters to be learned""" 

svclassifier = SVC(kernel ='poly', degree = 4)
"""
Adding polynomial features is simple to implement and can work with all sorts of
Machine Learning Algorithms.
•However at a low polynomial degree it cannot deal with very complex datasets,
•With a high polynomial degree it creates a huge number of features, making the
model too slow.
"""
svclassifier.fit(X_train, y_train)

"""
6. Making Predictions.
"""

y_pred = svclassifier.predict(X_test)

"""
# 7. Evaluating the Algorithm
"""

print('Poly karnel')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""
Gaussian Kernel
---------------
---------------
"""

X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']
y[y == 0] = -1
X = RobustScaler().fit_transform(X)

"""
This Scaler removes the median and scales the data according to the quantile 
range (defaults to IQR: Interquartile Range). The IQR is the range between 
the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
"""
# For `zero_division` control I need to RobustScalar and change label '0' to '1'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

"""
5. Fitting method of SVC class with Gaussian Kernel.
"""

svclassifier = SVC(kernel='rbf')

svclassifier.fit(X_train, y_train)

"""
# 6. Making Predictions.
"""

y_pred = svclassifier.predict(X_test)

"""
# 7. Evaluating the Algorithm
"""

print('Gaussian karnel')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""
Gamma is used when we use the Gaussian RBF kernel. if you use linear or
polynomial kernel then you do not need gamma only you need C hypermeter. 
Gamma is a hyperparameter which we have to set before training model.
Gamma decides that how much curvature we want in a decision boundary."""



"""
Sigmoid Kernel
--------------
--------------
"""
"""
5. Fitting method of SVC class with Sigmoid Kernel.
"""

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

"""
6. Making Predictions.
"""

y_pred = svclassifier.predict(X_test)

"""
# 7. Evaluating the Algorithm
"""

print('Sigmoid karnel')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
8. Comparison of Kernel Performance.

For this data:
If we compare the performance of the different types of kernels
we can clearly see that the Gaussian kernel performs the worst. 
This is due to the reason that Gaussian function returns two 
values, 0 and 1, therefore it is more suitable for binary classification 
problems sincewe forced it to be 0 and 1.However, in our case we had two output classes.
"""







