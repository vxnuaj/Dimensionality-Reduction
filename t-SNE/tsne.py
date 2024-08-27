import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nue.preprocessing import csv_to_numpy

data = csv_to_numpy('data/iris.csv')
X = data[:, 0:4]
labels = data[:, 4]

df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
df['Label'] = labels
df['Label'] = df['Label'].astype('category').cat.codes  

tsne = TSNE(n_components=2, random_state = 42)
X_reduced = tsne.fit_transform(X)

df[['Dimension 1', 'Dimension 2']] = X_reduced

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['Dimension 1'], df['Dimension 2'], c=df['Label'], cmap='viridis')
plt.colorbar(scatter, label='Label')
plt.title('t-SNE Results')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()