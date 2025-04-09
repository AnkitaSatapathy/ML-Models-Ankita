#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df[["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]].dropna()

X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]].values
y = df["Survived"].values  

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None
    def fit(self, X, y):
        """Train the SVM model using gradient descent"""
        n_samples, n_features = X.shape
        y_transformed = np.where(y == 0, -1, 1)  
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_transformed[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_transformed[idx]))
                    self.b -= self.learning_rate * y_transformed[idx]
    def predict(self, X):
        """Predict class labels"""
        predictions = np.dot(X, self.w) - self.b
        return np.where(predictions >= 0, 1, 0)

svm = CustomSVM()
svm.fit(X_train, y_train)

with open("svm_titanic_model.pkl", "wb") as f:
    pickle.dump({"model": svm, "scaler": scaler}, f)



