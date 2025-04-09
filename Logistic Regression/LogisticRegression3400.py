#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from sklearn.datasets import load_breast_cancer
import pickle

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=1000, reg_type=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_type = reg_type  
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(model)
            error = y_predicted - y

            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            if self.reg_type == "L1":
                dw += self.lambda_ * np.sign(self.weights)
            elif self.reg_type == "L2":
                dw += self.lambda_ * self.weights
            elif self.reg_type == "L1+L2":
                dw += self.lambda_ * (np.sign(self.weights) + self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """ Returns the probability of the sample being in class 1 """
        model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(model)

    def predict(self, X, threshold=0.6):
        """ Returns class labels based on a probability threshold """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

data = load_breast_cancer()
X = data.data[:, np.newaxis, 0]  
y = data.target

X = (X - np.min(X)) / (np.max(X) - np.min(X))  

model = LogisticRegressionCustom(reg_type="L2", lambda_=0.01)
model.fit(X, y)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[ ]:




