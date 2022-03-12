#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt


# In[24]:


file = 'ML_lab_testfile.txt'


# In[25]:


def Power(base, exp):
    result = 1
    while exp:
        if exp & 1:   #odd number
            result *= base
        exp = int(exp/2)     #/2
        base *= base
    return result


# In[26]:


def Transpose(matrix):  #2*3
    T_matrix=[]
    m = len(matrix) #3
    n = len(matrix[0])    #2
    for i in range(n):  #2
        T_matrix.append([matrix[j][i] for j in range(m)] ) 
    return np.array(T_matrix)


# In[27]:


def PrintMatrix(matrix):
    for row in matrix:
        print(row)


# In[28]:


def IdentityMatrix(n):
    I=[]
    for i in range(n):
        I.append([0] * n)
        I[i][i] = float(1)
    return np.array(I)


# In[29]:


def CalculateError(error_matrix):
    result = sum([error[0] * error[0] for error in error_matrix])
    return result


# In[30]:


def Inverse(matrix):
    size = len(matrix)
    In = np.eye(size,size)
    #print(size)
    for i in range(size):
        for j in range(size):
            if i != j:
                scalar = float(matrix[j][i]/matrix[i][i])
                for k in range(size):
                    matrix[j][k] -= scalar* matrix[i][k]
                    In[j][k] -= scalar* In[i][k]
    for i in range(size):
        ratio = matrix[i][i]
        for j in range(size):
            matrix[i][j] = matrix[i][j]/ratio
            In[i][j] = In[i][j]/ratio
    return In     


# In[31]:


def Formula(x, coef):
    result = np.zeros(x.shape)
    for i in range(len(coef)):
        result += coef[i][0] * np.power(x, len(coef) - i - 1)
    return result


# In[47]:


def PlotResult(coef, l_x, r_x, X , Y, type):
    x = np.arange(l_x, r_x, 0.1)
    y = Formula(x, coef)
    plt.subplot(2, 1, type)
    plt.plot(x, y)
    plt.scatter(X, Y, c='r', s=10, edgecolors='k')


# In[48]:


X=[]
Y=[]
with open(file) as f:
    for line in f.readlines():
        X.append(float(line.split(',',1)[0])) 
        Y.append(float(line.split(',',1)[1]))


# In[49]:


#main function
# Ax = b  n=3 [X^0,X^1,X^2]
case_num = int(input('number of case'))
for i in range(case_num):
    n = p = int(input('polynomial base: '))
    l = float(input('lambda: '))
    A = np.ones((len(X),n))
    b = []
    for i in range(n):
        for j in range(len(X)):
            A[j][i]=float(Power(X[j],p-1))
        p=p-1
    A = np.array(A)
    b = np.array(Y).reshape(len(X), 1)
    
    # for LSE
    ATA = Transpose(A).dot(A)
    ATA_lI = ATA + IdentityMatrix(len(ATA))*l 
    ATb = Transpose(A).dot(b)
    
    Inver = Inverse(ATA_lI)
    coef_LSE = Inver.dot(ATb)     #(ATA_lI)-1 ATb
    
    error_matrix = A.dot(coef_LSE) -b    # A (ATA_lI)-1 ATb -b
    error = CalculateError(error_matrix)
    
    print('LSE:\nFitting line: ', end='')
    for i in range(len(coef_LSE)):
        if i==0:
            print(coef_LSE[i][0], end='')
            if i != len(coef_LSE) - 1:
                print(f'X^{len(coef_LSE) - i - 1}', end='')
        else:
            if coef_LSE[i][0]>=0:
                print(' +',coef_LSE[i][0], end='')
                if i != len(coef_LSE) - 1:
                    print(f'X^{len(coef_LSE) - i - 1}', end='')
            else:
                print('',coef_LSE[i][0], end='')
                if i != len(coef_LSE) - 1:
                    print(f'X^{len(coef_LSE) - i - 1}', end='')
    print('\nTotal error:' ,error , '\n')
    
    #for Newton1
    coef_Newton =Inverse(ATA).dot(ATb)
    error_matrix = A.dot(coef_Newton)-b
    error = CalculateError(error_matrix)
    
    print('Newton\'s Method:\nFitting line: ', end='')
    for i in range(len(coef_Newton)):
        if i==0:
            print(coef_Newton[i][0], end='')
            if i != len(coef_LSE) - 1:
                print(f'X^{len(coef_Newton) - i - 1}', end='')
        else:
            if coef_Newton[i][0]>=0:
                print(' +',coef_Newton[i][0], end='')
                if i != len(coef_Newton) - 1:
                    print(f'X^{len(coef_Newton) - i - 1}', end='')
            else:
                print('',coef_Newton[i][0], end='')
                if i != len(coef_Newton) - 1:
                    print(f'X^{len(coef_Newton) - i - 1}', end='')
    print('\nTotal error:', error)
    
    PlotResult(coef_LSE, -6, 6, X, Y, 1)
    PlotResult(coef_Newton, -6, 6, X, Y, 2)
    plt.show()


