#!/usr/bin/env python
# coding: utf-8

# In[11]:


from Precode import *
import numpy
import pandas as pd
import random
from random import randrange
import matplotlib.pyplot as plt
print(pd.__version__)

data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S1('2289') 



# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[4]:


#Euclidean Distance Calculation
'''def Euc_dist(pt1,pt2):
    sum=0
    for i in range(len(pt1)):
        sum+=(pt1[i]-pt2[i])**2
    return (sum)**0.5
'''
#Initialization Part
def Cluster_centroids(k,data):
    centroid = [[]]*k
    for i in range(k):
        centroid[i]=list(data[randrange(len(data))])
    return np.asarray(centroid)


# In[5]:


from sklearn.cluster import KMeans
centroids=[]
inertia=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i,init=Cluster_centroids(i,data),n_init=1,random_state=0)
    kmeans.fit(data)
    centroids.append(kmeans.cluster_centers_)
    inertia.append(kmeans.inertia_)
print(centroids,end="")
print(inertia)


# In[ ]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=k1,init=i_point1,n_init=1,random_state=0)
kmeans.fit(data)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
print(centroids)
print(inertia)


# In[6]:


def plot(inertia):
    K_x=range(2,11,1)
    plt.plot(K_x,inertia)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Value of Objective Function')
    plt.title('Objective function vs No. of Clusters (k)')
    plt.show()
            
            
    



# In[12]:


plot(inertia)


# In[ ]:




