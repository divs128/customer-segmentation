import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("credit.csv")
print(data.head())
data = data.drop(["CUST_ID"], axis=1)
print(data.isnull().sum())
data["MINIMUM_PAYMENTS"].fillna(data["MINIMUM_PAYMENTS"].mean(skipna=True), inplace=True)
data["CREDIT_LIMIT"].fillna(data["CREDIT_LIMIT"].mean(skipna=True), inplace=True)
df = data.copy()
scaler = StandardScaler()
x = scaler.fit_transform(df)

kmeans = KMeans(5)
kmeans.fit(x)
labels = kmeans.labels_
clusters = pd.concat([data, pd.DataFrame({"Cluster": labels})], axis=1)
print(clusters.head())
colors = {0:"red", 1:"yellow", 2:"blue", 3:"green", 4:"pink"}
pca = PCA()
principal_components = pca.fit_transform(x)
x, y = principal_components[:,0], principal_components[:,1]
df = pd.DataFrame({"X": x, "Y": y, "Labels": labels})
groups = df.groupby(labels)

fig, ax = plt.subplots(figsize=(15, 10)) 

for i, j in groups:
    ax.plot(j.X, j.Y, marker='o', linestyle='', ms=5, color=colors[i], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.set_title("Credit Card Customer Segmentation")
plt.show()
