#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
from sklearn.datasets import load_diabetes
import pickle

class SimpleLinearRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = 0
        self.bias = 0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = self.weight * X + self.bias
            error = y_pred - y

            dW = (2/n) * np.dot(X, error)
            dB = (2/n) * np.sum(error)

            self.weight -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

    def predict(self, X):
        return self.weight * X + self.bias


diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  
y = diabetes.target


model = SimpleLinearRegressionCustom()
model.fit(X.flatten(), y)


with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[ ]:




