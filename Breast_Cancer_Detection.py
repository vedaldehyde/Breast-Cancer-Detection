## Importing the liabraries

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

## Importing the dataset
df=pd.read_csv('C:\\Users\\Vedant\\Desktop\\Data\\Breast_cancer_data.csv')
df.head()
df.tail()

## Checking the presence of NaN values
df.isnull().sum()

## Checking whether the dataset is balanced or not
sns.countplot(df["diagnosis"])

## Selecting best features for model training
sns.heatmap(data=df.corr(),annot=True)

## Graphical Analysis
sns.pairplot(df)

X=df.iloc[:,[0,2,3]] # Independent Variable
X

y=df.iloc[:,-1] # dependent Variable
y

## Splitting the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

## 1. Using K Nearest Neighbours
knnclassifier=KNeighborsClassifier(n_neighbors=6)
knnclassifier.fit(X_train,y_train)

**Predicting Accuracy**
y_pred1=knnclassifier.predict(X_test)
y_pred1

**Computing Accuracy Score**
knn=accuracy_score(y_test,y_pred1)*100
print("The accuracy for K Nearest Neighbours is:",knn,"%")

## 2. Using Decission Tree Classifier
dtclassifier=DecisionTreeClassifier(criterion='gini',random_state=50)
dtclassifier.fit(X_train,y_train)
y_pred2=dtclassifier.predict(X_test)
y_pred2

decission=accuracy_score(y_test,y_pred2)*100
print("The accuracy for Decission Tree Classifier is:",decission,"%")

## 3.  Using Support Vector Classifier
svcclassifier=SVC(kernel='rbf',random_state=0)
svcclassifier.fit(X_train,y_train)
y_pred3=svcclassifier.predict(X_test)
y_pred3
support=accuracy_score(y_test,y_pred3)*100
print("The accuracy for Support Vector Classifier is:",support,"%")

## 4. Using Random Forest
rfclassifier=RandomForestClassifier(criterion='entropy',random_state=50)
rfclassifier.fit(X_train,y_train)
y_pred4=rfclassifier.predict(X_test)
y_pred4
forest=accuracy_score(y_test,y_pred4)*100
print("The accuracy for Random Forest is:",forest,"%")

## 5. Using Logistic Regression
lrclassifier=LogisticRegression(random_state=50)
lrclassifier.fit(X_train,y_train)
y_pred5=lrclassifier.predict(X_test)
y_pred5
logistic=accuracy_score(y_test,y_pred5)*100
print("The accuracy for Logistic Regression is:",logistic,"%")

## The highest accuracy is: **91.22807017543859 %** for Random Forest

## Making Predictions
mean_radius=float(input('Enter mean radius of tumour'))
mean_perimeter=float(input('Enter mean perimeter of tumour'))
mean_area=float(input('Enter mean area of tumour'))


finalpred=rfclassifier.predict([[mean_radius,mean_perimeter,mean_area]])
if finalpred==1:
    print('Tumour is Malignant')
else:
    print('Tumour is Benign')
