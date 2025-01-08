import kagglehub
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("kartik2112/fraud-detection")
print("Path to dataset files:", path)

dataset = pd.read_csv(f'{path}/fraudTest.csv')

print(dataset.dtypes)

for column in dataset.select_dtypes(include=['object']).columns:
    try:
        dataset[column] = pd.to_datetime(dataset[column], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        dataset[column] = dataset[column].astype(np.int64) // 10**9
    except Exception:
        pass

numeric_columns = dataset.select_dtypes(include=[np.number]).columns
X = dataset[numeric_columns].values

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()
plt.show()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plt.plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2)
plt.show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 1)], mappings[(3, 2)]), axis=0)
frauds_orig = sc.inverse_transform(frauds)
frauds_orig = frauds_orig[:, 0].astype("int64")
print(frauds_orig)
