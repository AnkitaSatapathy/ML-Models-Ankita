#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        from sklearn.tree import DecisionTreeRegressor
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = [DecisionTreeRegressor(max_depth=max_depth) for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees:
            indices = np.random.choice(len(X), len(X), replace=True)  
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)  

model = SimpleRandomForest(n_trees=10, max_depth=5)
model.fit(X_train, y_train)

with open("california_housing_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)


# In[ ]:




