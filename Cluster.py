# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans  
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

dataframe = pd.read_csv("covid_19_data.csv", index_col = "SNo")

print(dataframe.shape)

df = dataframe.groupby(['Country/Region']).sum()

print(df.shape)

kmeans_model = KMeans(n_clusters = 5,init = "random").fit(df)

labels = kmeans_model.labels_

print(kmeans_model.cluster_centers_)

length = {i: np.where(kmeans_model.labels_ == i)[0] for i in range(kmeans_model.n_clusters)}
print (length)

'''pca = PCA(2)
plot_columns = pca.fit_transform(df)
plt.scatter(x = plot_columns[:,0], y = plot_columns[:,1], c = df['Confirmed'])
plt.show()'''
