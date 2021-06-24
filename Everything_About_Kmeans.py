# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:40:05 2021

@author: asabi
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns 
import matplotlib.pyplot as plt 
from pydataset import data

data = {'x': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50, 57,59,52,65, 47,49,48,35,33,44,45, 38,43,51,46],'y': [79,51,53, 78,59,74,73,57, 69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14, 12,20,5,29, 27,8,7]  }
data  
df = pd.DataFrame(data, columns=['x','y'])
print (df)
df.head()
df.mean()
df.max()
df.min()
print(df)

#%%kmeans: (%% -- seperator) --- need to specify clusters before
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(df) #specified here
centeriods=kmeans.cluster_centers_
print(centeriods) #average values of the cluster
dir(kmeans)
kmeans.n_clusters
kmeans.labels_ #number indicates cluster number
#all these together now to plot:
plt.scatter(df['x'],df['y'],c= kmeans.labels_.astype(float),s=50, alpha=.5) #alpha -> transparency, s is size
plt.scatter(centeroids[:,0],centeroids[:,1],c='red',s=100,marker='D') #D=Diamond shape
plt.show();  
  
#now with 4 clusters:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4).fit(df) #specified here
centeriods=kmeans.cluster_centers_
print(centeriods) #average values of the cluster
dir(kmeans)
kmeans.n_clusters
kmeans.labels_ #number indicates cluster number
#all these together now to plot:
plt.scatter(df['x'],df['y'],c= kmeans.labels_.astype(float),s=50, alpha=.5) #alpha -> transparency, s is size
plt.scatter(centeroids[:,0],centeroids[:,1],c='red',s=100,marker='D') #D=Diamond shape
plt.show();  




#%%
#trying on MTCars
import pandas as pd
from pydataset import data
mtcars=data('mtcars') 
mtcars.columns
mtcarsData=mtcars.copy()
id(mtcars)
id(mtcarsData)

#Selecting certain features based on which clustering is done 
kmeans = KMeans(n_clusters=3).fit(mtcarsData)
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_ 
#Cluster item Count 1
mtcars.groupby(kmeans.labels_).aggregate({'mpg':[np.mean,'count'],'wt':np.mean})
#Cluster item counter type 2
mtcarsData['kmean'] = kmeans.labels_
mtcarsData['kmean'].value_counts()


#items (column have different scales)
mtcarsData.min()
mtcarsData.max()
#WE WOULD NEED SCALING!

#%% SCALING mean 0, std =1 --Z score based
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() #Zscore Normalization
mtcarsScaledData=scaler.fit_transform(mtcarsData)
mtcarsScaledData[:5] #(values between -3 to +3)
np.min(mtcarsScaledData[:,1])
np.max(mtcarsScaledData[:,1])
kmeans =KMeans(init='random', n_init=3,max_iter=10,random_state=42,n_clusters=3).fit(mtcarsScaledData)
#max_iter --- max number of runs for kmeans algo


kmeans.n_iter_
kmeans.labels_
kmeans.inertia_
mtcarsData.groupby(kmeans.labels_).mean()
#or
mt=kmeans.labels_
mtcarsData.groupby([mt])['mpg','wt'].mean()

#now dendrogram
from sklearn.neighbors import DistanceMetric
from scipy.cluster.hierarchy import dendrogram, linkage
#Linkage Matrix
Z=linkage(mtcarsScaledData, method='ward')
dist=DistanceMetric.get_metric('euclidean')
dist
dist.pairwise(mtcarsScaledData)
#plotting
dendro=dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
#largest vertical line without any line passing