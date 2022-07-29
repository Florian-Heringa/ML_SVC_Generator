import numpy as np
from numpy.random import uniform as uf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pandas as pd
import seaborn as sns

clf = SVC(kernel='linear')

def gen_datacloud(x0, y0, radius, N):

    r = radius * np.random.rand(N)
    theta = 360.0 * np.random.rand(N)

    x = r * np.sin(theta)
    y = r * np.cos(theta)

    x += x0
    y += y0

    return x, y
train_test_split()

# x1, y1 = gen_datacloud(2, 4, 6, 100)

# plt.scatter(x1, y1)
# plt.plot(2, 4, 'ro')
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.gca().set_aspect('equal')
# plt.show()
N = 1000

x1, y1 = gen_datacloud(uf(low=0, high=10), uf(low=0, high=10), uf(low=0, high=5), N)
cl1 = np.zeros(x1.shape)

x2, y2 = gen_datacloud(uf(low=-10, high=0), uf(low=-10, high=0), uf(low=0, high=5), N)
cl2 = np.ones(x2.shape)

xs = np.concatenate((x1, x2))
ys = np.concatenate((y1, y2))
cl = np.concatenate((cl1, cl2))

print(xs.shape, ys.shape, cl.shape)

data = np.hstack((cl[:,None], xs[:,None], ys[:,None]))

print(data.shape)

df = pd.DataFrame(data)
df.columns = ['class', 'x', 'y']
df = df.astype({'class': int})

print(df.head())

sns.catplot(x='x', y='y', data = df)
plt.show()