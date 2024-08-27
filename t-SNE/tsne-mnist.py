import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nue.preprocessing import csv_to_numpy

RANDOM_STATE = 1

index = 10000
data = csv_to_numpy('data/fashion-mnist_train.csv')
X = data[:, 1:]
labels = data[:, 0]


df = pd.DataFrame(X[0:index], columns = [f'{i}' for i in range(X.shape[1])])
df['Label'] = labels[0:index]
df['Label'] = df['Label'].astype('category').cat.codes

print('TSNE Fitting.')
tsne = TSNE(n_components = 2, perplexity = 10, random_state = RANDOM_STATE)
X_reduced = tsne.fit_transform(X[0:index, :])

df[['Dimension 1', 'Dimension 2']] = X_reduced

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['Dimension 1'], df['Dimension 2'], c=df['Label'], cmap='viridis')
plt.colorbar(scatter, label='Label')
plt.title('t-SNE Results')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()