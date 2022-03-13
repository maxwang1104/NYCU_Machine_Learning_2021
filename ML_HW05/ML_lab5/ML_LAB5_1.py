#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


# In[2]:


def Load_data():
    X = []
    Y = []
    with open("input.data",'r') as file:
        for line in file:
            x,y = line.split(' ')
            X.append(float(x))
            Y.append(float(y))
    return np.array(X).reshape(-1,1) , np.array(Y).reshape(-1,1)        


# In[3]:


def Rational_quadratic_kernel(sigma,alpha,l,xi,xj):
    dist = np.sum(xi ** 2,axis=1).reshape(-1,1) + np.sum(xj ** 2,axis=1) - 2*xi.dot(np.transpose(xj))
    kernel = (sigma ** 2) * (1 + dist / (2 * alpha * (l ** 2))) ** (-1 * alpha)
    return kernel   


# In[4]:


def Negative_marginal_likelihood(theta):
    theta.ravel()
    C = Rational_quadratic_kernel(theta[0],theta[1],theta[2],X,X) + np.identity(len(X))*(1/beta)
    res = 0.5*np.log(np.linalg.det(C)) + 0.5*np.transpose(Y).dot(np.linalg.inv(C)).dot(Y) + len(X)/2*np.log(2*math.pi)
    return res.ravel()


# In[5]:


def Gaussian_process(X,Y,X_star,beta,sigma,alpha,l):
    kernel = Rational_quadratic_kernel(sigma,alpha,l,X,X)
    kernel_1star = Rational_quadratic_kernel(sigma,alpha,l,X,X_star)
    kernel_2star = Rational_quadratic_kernel(sigma,alpha,l,X_star,X_star)
    C = kernel + np.identity(len(X))*(1/beta)
    inver_C = np.linalg.inv(C)
    
    mean_star = np.transpose(kernel_1star).dot(inver_C).dot(Y)
    k_star = kernel_2star + (1/beta)
    var_star = k_star - np.transpose(kernel_1star).dot(inver_C).dot(kernel_1star)
    #print(mean_star.shape,k_star.shape,var_star.shape)
    
    #visualization
    plt.plot(X_star, mean_star)
    plt.scatter(X, Y, s=10)
    
    interval = 1.95 * np.sqrt(np.diag(var_star))
    X_star = X_star.reshape(-1)
    mean_star = mean_star.reshape(-1)

    plt.plot(X_star, mean_star + interval,color = "green")
    plt.plot(X_star, mean_star - interval,color = "green")
    plt.fill_between(X_star, mean_star + interval, mean_star - interval,alpha=0.3)

    #plt.title(f'sigma: {sigma:.5f}, alpha: {alpha:.5f}, length scale: {l:.5f}')
    plt.xlim(-60, 60)
    plt.show()


# In[6]:


if __name__ == '__main__':
    X,Y = Load_data()
    X_perdict = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    beta = 5
    sigma = 1
    alpha = 1
    l = 1
    Gaussian_process(X, Y, X_perdict, beta, sigma, alpha, l)
    
    theta = [1,1,1]
    res = minimize(Negative_marginal_likelihood, theta)
    sigma_ = res.x[0]
    alpha_ = res.x[1]
    l_ = res.x[2]
    Gaussian_process(X, Y, X_perdict, beta, sigma_, alpha_, l_)
    


# In[ ]:




