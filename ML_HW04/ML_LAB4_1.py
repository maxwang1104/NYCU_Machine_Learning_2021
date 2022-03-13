#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import time


# In[2]:


def UnivariateGaussianDataGenerator(mean, std_dev):
    return mean + std_dev * (sum(np.random.uniform(0, 1, 12)) - 6)


# In[3]:


def GenerateDataPointSet(n, mx, vx, my, vy, label):
    data_x = []
    data_y = []
    X = []
    y = []
    for i in range(n):
        xi = UnivariateGaussianDataGenerator(mx, math.sqrt(vx))
        yi = UnivariateGaussianDataGenerator(my, math.sqrt(vy))
        data_x.append(xi)
        data_y.append(yi)
        X.append([xi, yi, 1])
        y.append([label])
    return data_x, data_y, X, y


# In[4]:


def sigmoid(A):
    matrix = []
    for i in range(0, len(A)):
        temp = []
        for j in range(0, len(A[0])):
            temp.append(1.0 / (1.0 + np.exp(-1.0 * A[i][j])))
        matrix.append(temp)
    return np.array(matrix)


# In[5]:


def Difference(vector1, vector2):
    break_or_not = True
    for (i, j) in zip(vector1, vector2):
        if abs(i - j) > (abs(j) * 0.075):
            break_or_not = False
            break
    return break_or_not


# In[6]:


def I(n):
    return np.identity(n)


# In[7]:


def Inverse(matrix):
    return np.linalg.inv(matrix)


# In[8]:


def Determinant(A):
    return np.linalg.det(A)


# In[9]:


def Transpose(matrix):  
    return np.transpose(matrix)


# In[10]:


def Gradient_decent(X,y):
    learning_rate = 0.05
    pre_w = np.zeros(3).reshape(3,1)
    w = np.zeros(3).reshape(3,1)
    XT = Transpose(X)
    while True:
        Xw = X.dot(pre_w)
        gradient = XT.dot(y - sigmoid(Xw))
        w = pre_w + gradient * learning_rate
        if Difference(w,pre_w):
            break
        pre_w = w
    return w


# In[55]:


def MatrixD(X,w):
    D = []
    for i in range(0, len(X)):
        temp = []
        for j in range(0, len(X)):
            if i == j:
                temp1 = -1.0 * (X[i][0] * w[0][0] + X[i][1] * w[1][0] + X[i][2] * w[2][0])
                temp2 = np.exp(temp1)
                if math.isinf(temp2):
                    temp2 = np.exp(700)
                temp.append(temp2 / ((1 + temp2) ** 2))
            else:
                temp.append(0.0)
        D.append(temp)
    return np.array(D)


# In[50]:


def Newtons_method(X, y):
    pre_w = np.zeros(3).reshape(3,1)
    w = np.zeros(3).reshape(3,1)
    n = len(X)
    XT = Transpose(X)
    while True:
        Xw = X.dot(pre_w)
        D = MatrixD(X,w)
        Hessian = Transpose(X).dot(D.dot(X))
        gradient = XT.dot(y - sigmoid(Xw))
        
        if Determinant(Hessian) == 0:
            w = pre_w + (gradient * learning_rate)
        else:
            w = pre_w + (Inverse(Hessian).dot(gradient))
        if Difference(Transpose(w)[0],Transpose(pre_w)[0]):
            break
        pre_w = w
    return w


# In[51]:


def SubplotResult(X, w, y, class1_x, class1_y, class2_x, class2_y, title, separate, subplot_idx):
    if title == 'Ground truth' :
        plt.subplot(subplot_idx)
        plt.title(title)
        plt.scatter(class1_x, class1_y, c='r')
        plt.scatter(class2_x, class2_y, c='b')
    else:
        confusion_matrix = [[0, 0], [0, 0]]
        predict = sigmoid(X.dot(w))
        predict_class1_x = []
        predict_class1_y = []
        predict_class2_x = []
        predict_class2_y = []
        for i in range(0, len(predict)):
            if y[i][0] == 0:
                if predict[i][0] < 0.5:
                    predict_class1_x.append(X[i][0])
                    predict_class1_y.append(X[i][1])
                    confusion_matrix[0][0] += 1
                else:
                    predict_class2_x.append(X[i][0])
                    predict_class2_y.append(X[i][1])
                    confusion_matrix[0][1] += 1
            if y[i][0] == 1:
                if predict[i][0] < 0.5:
                    predict_class1_x.append(X[i][0])
                    predict_class1_y.append(X[i][1])
                    confusion_matrix[1][0] += 1
                else:
                    predict_class2_x.append(X[i][0])
                    predict_class2_y.append(X[i][1])
                    confusion_matrix[1][1] += 1
        print(f'{title}:\n')
        print('w:')
        print(w)
        print('\nConfusion Matrix:')
        print('\t\tPredict cluster 1 Predict cluster 2')
        print(f'Is cluster 1\t\t{confusion_matrix[0][0]}\t\t{confusion_matrix[0][1]}')
        print(f'Is cluster 2\t\t{confusion_matrix[1][0]}\t\t{confusion_matrix[1][1]}')
        sens = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        spec = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        print(f'\nSensitivity (Successfully predict cluster 1): {sens}')
        print(f'Specificity (Successfully predict cluster 2): {spec}')
        if separate:
            print('\n----------------------------------------')
        plt.subplot(subplot_idx)
        plt.title(title)
        plt.scatter(predict_class1_x, predict_class1_y, c='r')
        plt.scatter(predict_class2_x, predict_class2_y, c='b')


# In[52]:


#if __name__ == '__main__':
N = 50
mx1 = my1 = 1
mx2 = my2 = 3
vx1 = vy1 = 2
vx2 = vy2 = 4

class1_x, class1_y, X, y = GenerateDataPointSet(N, mx1, vx1, my1, vy1, 0)
class2_x, class2_y, tempX, tempy = GenerateDataPointSet(N, mx2, vx2, my2, vy2, 1)
X += tempX
X = np.array(X).reshape(2*N,3)
y += tempy
y = np.array(y).reshape(2*N,1)


SubplotResult(None, None, None, class1_x, class1_y, class2_x, class2_y, 'Ground truth', None, 131)
gradient_w = Gradient_decent(X, y)
SubplotResult(X, gradient_w, y, class1_x, class1_y, class2_x, class2_y, 'Gradient descent', True, 132)
Newton_w = Newtons_method(X, y)
SubplotResult(X, Newton_w, y, class1_x, class1_y, class2_x, class2_y, 'Newton\'s method', False, 133)
plt.tight_layout()
plt.show()


# In[ ]:




