import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

pred10 = np.loadtxt("pred10.csv", delimiter=',', dtype=str)
labels = np.loadtxt("val.csv", delimiter=',', dtype=str)

data = []
find = np.zeros(10)
vars = np.zeros(10)

for pred, label in tqdm(zip(pred10, labels)):
    preds = pred[1].split('_')
    label = label[1]
    weights = pred[2].split('_')
    weights = softmax([float(i) for i in weights])

    if label in preds:
        find[preds.index(label)] += 1
        vars[preds.index(label)] += np.var(weights)
        data.append([label, preds, weights])

df = pd.DataFrame(data)
print(df)

print(labels.shape)
print(find)
print(np.sum(find[1:]) / np.sum(find) * 100)
print(vars / find)