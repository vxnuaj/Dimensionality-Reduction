import numpy as np
from pca import PCA
from nue.preprocessing import csv_to_numpy, x_y_split

# --- IRIS DATASET
'''
data = csv_to_numpy('data/iris.csv')
X, Y = x_y_split(data, y_col = 'last')
print(f"Iris Orig, 4 Features (Top 4 Rows):\n{X[:4]}\n")

pca = PCA(n_components=3)
X_fit = pca.fit_transform(X = X)
print(f"Iris Reduced, 3 Features (Top 4 Rows):\n{X_fit[:4]}") 
'''

# --- MNIST DATASET ---

data =csv_to_numpy('data/fashion-mnist_train.csv')
X, Y = x_y_split(data, y_col = 'first')

pca = PCA(n_components = 750)
X_fit = pca.fit_transform(X = X)
print(f"FASHION MNIST Original.\nShape:{X.shape}\nTop 4 Rows:\n{X[:4]}\n")
print(f"FASHION MNIST Reduced to 750 features.\nShape:{X_fit.shape}\nTop 4 Rows:\n{X_fit[:4]}")