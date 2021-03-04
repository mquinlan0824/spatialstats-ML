# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:04:20 2021
Topic: k-means clustering on dummy dataset
@author: Michael Quinlan
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

# Create dummy dataset containing 350 2-dimensional data points.
points = np.vstack(((np.random.randn(150, 2) * 0.7 + np.array([2, 0])),
                  (np.random.randn(100, 2) * 0.3 + np.array([-0.5, 1.0])),
                  (np.random.randn(100, 2) * 0.5 + np.array([0.0, -1.0]))))


# This is a function to be fitted to the data
kmean_cluster = cluster.KMeans(n_clusters=3)  # To assign number of clusters to detect 
labels = kmean_cluster.fit_predict(points)  # Perform clustering on points and return cluster labels
# NOTE: cluster labels will be numeric: 0, 1, 2

# Visualize the clusters by using the 'labels'
plt.figure(figsize=(10,8))
plt.scatter(points[:, 0], points[:, 1], s=40, c=labels, cmap='Paired', marker='x', label='training data')
plt.title('Dummy Data')
plt.legend()

