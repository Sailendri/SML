#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Precode2 import *
import numpy
import matplotlib.pyplot as plt

data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S2('2289') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[30]:


initial_center=[i_point1]
while len(initial_center)<k1:
    max_avg_dist=-1
    calc_center=None
    
    for pt in data:
        #calculate distance to centers
        distances_to_centers = np.linalg.norm(pt - np.array(initial_center), axis=1)
        avg_dist=np.mean(distances_to_centers)
        
        if avg_dist > max_avg_dist and not np.any(initial_center==pt):
            max_avg_dist=avg_dist
            calc_center=pt
    if calc_center is not None:
        initial_center.append(calc_center)
cl_centroids=np.array(initial_center)
print(cl_centroids)


# In[31]:


centroids=[]
inertia=[]
#for i in range(2,11):
kmeans=KMeans(n_clusters=k1,init=cl_centroids,random_state=0)
kmeans.fit(data)
centroids.append(kmeans.cluster_centers_)
inertia.append(kmeans.inertia_)
print(centroids,end="")
print(inertia)


# In[23]:


from sklearn.cluster import KMeans
initial_center_2=[i_point2]

while len(initial_center_2)<k2:
    max_avg_dist_2=-1
    calc_center=None
    
    for pts in data:
        distances_to_centers_2 = np.linalg.norm(pts - np.array(initial_center_2), axis=1)
        avg_dist_2=np.mean(distances_to_centers_2)
      
        if avg_dist_2 > max_avg_dist_2 and not np.any(initial_center_2==pts):
            max_avg_dist_2=avg_dist_2
            calc_center=pts
            
    if calc_center is not None:
        initial_center_2.append(calc_center)

cl_centroids_2=np.array(initial_center_2)
print(cl_centroids_2)



# In[24]:


centroids_2=[]
inertia_2=[]
#for i in range(2,11):
kmeans_2=KMeans(n_clusters=k2,init=cl_centroids_2,random_state=0)
kmeans_2.fit(data)
centroids_2.append(kmeans_2.cluster_centers_)
inertia_2.append(kmeans_2.inertia_)
print(centroids_2,end="")
print(inertia_2)


# In[4]:


from sklearn.cluster import KMeans
centroids=[]
inertia=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(data)
    centroids.append(kmeans.cluster_centers_)
    inertia.append(kmeans.inertia_)
print(centroids,end="")
print(inertia)


# In[5]:


def plot(inertia):
    K_x=range(2,11,1)
    plt.plot(K_x,inertia)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Value of Objective Function')
    plt.title('Objective function vs No. of Clusters (k)')
    plt.show()


# In[6]:


plot(inertia)


# In[ ]:




