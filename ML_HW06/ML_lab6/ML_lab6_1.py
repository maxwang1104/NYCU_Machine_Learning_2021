#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy.spatial.distance import pdist, cdist
import numpy as np
from PIL import Image
import random
import os
import sys


# In[4]:


def load_png(filename):
    img = Image.open(filename)
    img = np.array(img.getdata()) #(10000,3)
    return img


# In[5]:


def Two_RBF_kernel(X, gamma_s, gamma_c): #img, spatial, color
    dist_c = cdist(X,X,'sqeuclidean') #(10000,10000)

    seq = np.arange(0,100)
    c_coord = seq
    for i in range (99):
        c_coord = np.hstack((c_coord, seq))
    c_coord = c_coord.reshape(-1,1)
    r_coord = c_coord.reshape(100,100).T.reshape(-1,1)
    
    X_s = np.hstack((r_coord, c_coord))
    dist_s = cdist(X_s,X_s,'sqeuclidean')

    RBF_s = np.exp(-gamma_s * dist_s)
    RBF_c = np.exp(-gamma_c * dist_c) #(10000,10000)
    k = np.multiply(RBF_s, RBF_c) #(10000,10000)
    
    return X_s, k


# In[6]:


def initial(k, mode):
    if mode == "random":
        centers = list(random.sample(range(0,10000), k))
    
    elif mode == "kmeans++":
        centers = []
        centers = list(random.sample(range(0,10000), 1))
        found = 1
        while (found<k):
            dist = np.zeros(10000)
            for i in range(10000):
                min_dist = np.Inf
                for f in range(found):
                    tmp = np.linalg.norm(X_spatial[i,:] - X_spatial[centers[f],:])
                    if tmp<min_dist:
                        min_dist = tmp
                dist[i] = min_dist
            dist = dist/np.sum(dist)
            idx = np.random.choice(np.arange(10000), 1, p=dist)
            centers.append(idx[0])
            found += 1
    # print(centers)
    return centers


# In[7]:


def initial_kernel_kmeans(K, data, mode):
    centers = initial(K, mode)

    N = len(data)
    cluster = np.zeros(N, dtype=int)
    for n in range(N):
        dist = np.zeros(K)
        for k in range(K):
            dist[k] = data[n,n] + data[centers[k],centers[k]] - 2*data[n,centers[k]]
        cluster[n] = np.argmin(dist)
    return cluster


# In[8]:


def construct_sigma_n(kernelj, cluster, cluster_k):
    ker = kernelj.copy()
    mask = np.where(cluster==cluster_k)
    sigma = np.sum(ker[mask])
    return sigma


# In[9]:


def construct_sigma_pq(C, K, kernel, cluster):
    pq = np.zeros(K)
    for k in range(K):
        ker = kernel.copy()
        for n in range(len(kernel)):
            if cluster[n]!=k:
                ker[n,:] = 0
                ker[:,n] = 0
        pq[k] = np.sum(ker)/C[k]/C[k]
    return pq


# In[10]:


def construct_C(K, cluster):
    C = np.zeros(K, dtype=int)
    for k in range(K):
        indicator = np.where(cluster==k, 1, 0)
        C[k] = np.sum(indicator)
    return C


# In[11]:


def clustering(K, kernel, cluster):
    N = len(kernel)
    new_cluster = np.zeros(N, dtype=int)
    C = construct_C(K, cluster)
    pq = construct_sigma_pq(C, K, kernel, cluster)
    for j in range(N):
        dist = np.zeros(K)
        for k in range(K):
            dist[k] += kernel[j,j] + pq[k]
            dist[k] -= 2/C[k] * construct_sigma_n(kernel[j,:], cluster, k)
        new_cluster[j] = np.argmin(dist)

    return new_cluster


# In[12]:


def kernel_kmeans(K, kernel, cluster, iter, mode):
    save_png(len(kernel), K, cluster, 0, mode)
    for i in range(1, iter+1):
        print("iter", i)
        new_cluster = clustering(K, kernel, cluster)
        if(np.linalg.norm((new_cluster-cluster), ord=2)<1e-2):
            break
        cluster = new_cluster
        save_png(len(kernel), K, cluster, i, mode)


# In[21]:


def save_png(N, K, cluster, iter, mode):
    colors = np.array([[255,97,0],[0,0,0],[107,142,35],[244,164,96],[95,0,135],[61,89,171]])
    result = np.zeros((100*100, 3))
    for n in range(N):
        result[n,:] = colors[cluster[n],:]
    
    img = result.reshape(100, 100, 3)
    img = Image.fromarray(np.uint8(img))
    #img.save(os.path.join('C:\\Users\\88692\\ML_lab6\\image', '%d-cluster'%K ,'%d.png'%iter))
    img.save(f'C:\\Users\\88692\\ML_lab6\\image\\3-cluster-image2\\{mode}_{K}-cluster {iter}.png')


# In[23]:


if __name__=='__main__':
    k = int(input("K-cluster"))
    mode = input("random or kmeans++")
    X_color = load_png("image2.png") #(10000, 3)
    X_spatial, fi_X = Two_RBF_kernel(X_color, 1 / (100 * 100), 1 / (255 * 255)) #spatial, color
    cluster = initial_kernel_kmeans(k, fi_X, mode)
    kernel_kmeans(k, fi_X, cluster, 1000, mode)


# In[ ]:




