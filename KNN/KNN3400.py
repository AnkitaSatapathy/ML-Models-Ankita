#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pickle
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url)

df = df.dropna()
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = df[features].values 
y = df["species"].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Store training data"""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Predict the class for each test sample"""
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        """Helper function to predict a single sample"""
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

knn = CustomKNN(k=5)
knn.fit(X_train, y_train)

with open("custom_knn_penguins_model.pkl", "wb") as f:
    pickle.dump(knn, f)


# In[ ]:




