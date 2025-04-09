#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes()
X = data.data[:, :4]  
y = (data.target > data.target.mean()).astype(int) 

feature_names = data.feature_names[:4]  
target_names = ["No Diabetes", "Diabetes"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf node

class DecisionTree:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                parent_entropy = self.entropy(y)
                left_entropy = self.entropy(y[left_indices])
                right_entropy = self.entropy(y[right_indices])

                gain = parent_entropy - (len(y[left_indices]) / len(y) * left_entropy +
                                         len(y[right_indices]) / len(y) * right_entropy)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(set(y)) == 1:
            return Node(value=np.argmax(np.bincount(y)))

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return Node(value=np.argmax(np.bincount(y)))

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])

dt = DecisionTree(max_depth=4)
dt.fit(X_train, y_train)

model_data = {"model": dt, "features": feature_names, "targets": target_names}

with open("custom_decision_tree.pkl", "wb") as f:
    pickle.dump(model_data, f)


# In[ ]:




