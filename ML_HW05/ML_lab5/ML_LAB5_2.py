#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from libsvm.svmutil import *
import time


# In[2]:


def read_file(filename):
    with open(filename,'r') as file:
        lines = file.readlines()
    lines = np.array( [line.strip().split(',') for line in lines] , dtype = 'float64')
    return lines


# In[3]:


def read_MINST():
    X_train = read_file("X_train.csv")   #5000 * 784
    X_test = read_file("X_test.csv")     #2500 * 784
    Y_train = read_file("Y_train.csv")   #5000 * 1
    Y_test = read_file("Y_test.csv")     #2500 * 1
    
    Y_train = Y_train.reshape(1,-1)
    Y_test = Y_test.reshape(1,-1)
    
    return X_train, X_test, Y_train, Y_test


# In[4]:


def SVM(kernel,param):
    print(kernel)
    param += " -q"
    param_ = svm_parameter(param)
    
    start = time.time()
    prob = svm_problem(Y_train[0],X_train)   #prob  = svm_problem(y, x)
    model = svm_train(prob,param_)        #model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(Y_test[0], X_test, model)  #p_label, p_acc, p_val = svm_predict(yt, xt, model)
    print("%0.2f sec" %(time.time()-start))
    print("p_acc",p_acc)
    print("\n")
    return p_acc


# In[5]:


def GridSearch_svm(kernel_type,param,fold):
    print(kernel_type,"\n")
    if fold !=0:
        param += " -v "+str(int(fold))
    param += " -q"
    param_ = svm_parameter(param)
    prob = svm_problem(Y_train[0],X_train)
    acc = svm_train(prob,param_)
    
    return acc


# In[6]:


def GridSearch(kernel_type):
    c = [1e-2,1e-1,1,10,100]   #cost
    #d = range(1,10)           #polynomial_kernel degree
    g = [1e-2,1e-1,1,10,100]   #polynomial_kernel and kbf gmma
    #r = range(-10,10,1)       #polynomial_kernel cof0 
    n = 3
    
    max_acc = 0
    max_param = 0
    param = 0
    if kernel_type == 'Linear': #-c
        for ci in c:
            param = " -c " + str(float(ci))
            acc = GridSearch_svm(kernel_type+param, "-t 0"+param, n)
            if acc > max_acc:
                max_acc = acc
                max_param = param
                
    elif kernel_type == 'Polynomail': # -c -r -g -d
        for ci in c:
            param = " -c " + str(float(ci))
            for ri in range(0,5,1):
                param_r = param + " -r " + str(int(ri))
                for gi in g:
                    param_g = param_r + " -g " + str(float(gi))
                    for di in range(2,5):
                        param_d = param_g + " -d " + str(int(di))
                        acc = GridSearch_svm(kernel_type + param_d, "-t 1" + param_d, n)
                        if acc > max_acc:
                            max_acc = acc
                            max_param = param_d
        

    elif kernel_type == 'RBF': #-c -g
        for ci in c:
            param = " -c " + str(float(ci))
            for gi in g:
                    param_g = param + " -g " + str(float(gi))
                    acc = GridSearch_svm(kernel_type + param_g, "-t 1" + param_g, n)
                    if acc > max_acc:
                        max_acc = acc
                        max_param = param_g
    print("#############")
    print("kerenl_type",kernel_type)
    print("Max acc",max_acc)
    print("Max param",max_param)
    print("#############")


# In[7]:


def Linear_kernel(xi,xj):
    return xi.dot(np.transpose(xj))


# In[8]:


def RBF_kernel(xi,xj,g):
    dist = np.sum(xi ** 2,axis=1).reshape(1,-1) + np.sum(xj ** 2,axis=1) -2 * xi.dot(np.transpose(xj))
    kernel = np.exp((-1 * g * dist))
    return kernel


# In[11]:


if __name__ == '__main__':
    
    X_train, X_test, Y_train, Y_test = read_MINST()

    Part 1
    SVM("Linear","-s 0 -t 0")  #-c 
    SVM("Polynomail","-s 0 -t 1") # -c -r -g -d
    SVM("RBF","-s 0 -t 2") #-c -g
    
    #Part 2
    GridSearch("Linear")
    GridSearch("Polynomail")
    GridSearch("RBF")
    
    
    #Part 3
    g = [1/784, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    c = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000]
    row, col = X_train.shape
    max_acc = 0.0
    g_best = 0
    linear_k = Linear_kernel(X_train, X_train)
    for gi in range(len(g)):
        rbf_k = RBF_kernel(X_train, X_train, -g[gi])
        user_k = linear_k + rbf_k
        user_k = np.hstack((np.arange(1, row+1).reshape(-1, 1), user_k))
        prob = svm_problem(Y_train[0], user_k, isKernel=True)
        for ci in range(len(c)):
            param_str = "-t 4 -c " + str(c[ci]) + " -v 3 -q"
            param_rec = "-t 4 -c " + str(c[ci]) + " -q"
            print("-g", g[gi], param_str)
            param = svm_parameter(param_str)
            val_acc = svm_train(prob, param)

            if val_acc > max_acc:
                max_acc = val_acc
                max_param = param_rec
                g_best = gi
    print("============================")
    print("Best Parameters:", " -g", g[g_best], max_param)
    print("Max accuracy:", max_acc)
    X_train, X_test, Y_train, Y_test = read_MINST()
    row, col = X_train.shape
    print("row",row)

    linear_k = Linear_kernel(X_train,X_train)
    rbf_k = RBF_kernel(X_train, X_train, -0.001)
    user_k = linear_k + rbf_k
    print("user",user_k.shape)
    print("np",np.arange(1, row+1).reshape(-1, 1))
    user_k = np.hstack((np.arange(1, row+1).reshape(-1, 1), user_k))
    prob = svm_problem(Y_train[0], user_k, isKernel=True)
    param = svm_parameter("-t 4 -g 0.001 -c 0.1 -q")
    model = svm_train(prob, param)

    row, col = X_test.shape
    linear_k = Linear_kernel(X_test, X_test)
    rbf_k = RBF_kernel(X_test, X_test, -0.001)
    user_k = linear_k + rbf_k
    print("user",user_k.shape)
    print("np",np.arange(1, row+1).reshape(-1, 1))
    user_k = np.hstack((np.arange(1, row+1).reshape(-1, 1), user_k))
    _, p_acc,_ = svm_predict(Y_test[0], user_k, model)
    print("p_acc",p_acc)
    print("\n")
    


# In[12]:




