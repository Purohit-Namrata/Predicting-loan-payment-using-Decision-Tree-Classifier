import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Decision-tree examle/Decision_Tree_Dataset.csv")
print(df)
X = df.values[:,1:5]   #first ':' represents rows and second ':' represents columns
Y = df.values[:,0]     
print(X)
print(Y)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
model=DecisionTreeClassifier(criterion='entropy',random_state=100)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred)   # here we get 300 predictions: because 300 out of 1000 is test data

print("Accuracy is ",accuracy_score(y_test,y_pred)*100)
