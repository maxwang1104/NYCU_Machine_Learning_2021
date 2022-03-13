#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import math
import time


# In[3]:


def UnivariateGaussianDataGenerator(mean, std_dev):
    return mean + std_dev * (sum(np.random.uniform(0, 1, 12)) - 6)


# In[4]:


def PolynomialBasisLinearModelDataGenerator(n, std_dev, w):
    x = np.random.uniform(-1, 1)
    y = sum([w[i] * (x ** i) for i in range(n)]) + UnivariateGaussianDataGenerator(0, std_dev)
    return float(x), float(y)


# In[5]:


def I(n):
    return np.identity(n)


# In[6]:


def Inverse(matrix):
    return np.linalg.inv(matrix)


# In[7]:


def Transpose(matrix):  
    return np.transpose(matrix)


# In[8]:


def PrintMatrix(matrix):
    for row in matrix:
        print(row)


# In[9]:


def SubplotResult(idx, title, x, y, m, a, lambda_inverse, err_var, ground_truth):
    plt.subplot(idx)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    function = np.poly1d(np.flip(m))
    x_curve = np.linspace(-2.0, 2.0, 30)
    y_curve = function(x_curve)
    plt.plot(x_curve, y_curve, 'k')
    if ground_truth:
        plt.plot(x_curve, y_curve + err_var, 'r')
        plt.plot(x_curve, y_curve - err_var, 'r')
    else:
        plt.scatter(x, y, s=10)
        y_curve_plus_var = []
        y_curve_minus_var = []
        X = np.zeros((1,n))
        for i in range(0, 30):    
            for j in range(0, n):
                X[0][j] = x_curve[i] ** j
            #print("X\n",X)
            #print("lambda_inverse\n",lambda_inverse)
            distance = (1 / a) + (X.dot(lambda_inverse).dot(Transpose(X)))[0][0]
            #print("distance ",distance)
            y_curve_plus_var.append(y_curve[i] + distance)
            y_curve_minus_var.append(y_curve[i] - distance)
        plt.plot(x_curve, y_curve_plus_var, 'r')
        plt.plot(x_curve, y_curve_minus_var, 'r')


# In[11]:


if __name__ == '__main__':
    #b = 1
    #n = 3
    #a_err = 3
    #w = [1,2,3]
    b = int(input("b = ")) #1
    n = int(input("n = ")) #4
    a_err = int(input("a = ")) #1
    w = [float(val.strip('[]')) for val in input('Input w: ').split(',')] #[1,2,3,4]
    cur_mean = np.zeros(n).reshape(n,1)
    pre_mean = np.zeros(n).reshape(n,1)
    x = []
    y = []
    cur_var_pred = 0
    pre_var_pred = 0

    num = 0
    lamda = np.zeros((n,n))
    C = np.zeros((n,n))
    START = time.time()

    while 1:
        new_y = [[0.0]]
        new_x, new_y[0][0] = PolynomialBasisLinearModelDataGenerator(n, math.sqrt(a_err), w)
        num += 1
        x.append(new_x)
        y.append(new_y[0][0])
        print(f'Add data point ({round(new_x,5)}, {round(new_y[0][0],5)}):')
        var = 0
        '''
        temp = np.zeros((1,len(x)))
        for i in range(len(x)):
            temp = [x[i] **j for j in range(0,n)]
            var += (y[i] - (temp*pre_mean)[0][0]) ** 2
        var /= num
        a = 1 / (0.00000001 if var == 0 else var)
        '''
        #print("a ",a)
        a = a_err

        X = np.zeros((1,n))  #1*4
        for i in range(n):
            X[0][i] = new_x **i


        if num ==1:
            lamda = a*Transpose(X).dot(X) + b*I(n)   # Λ = aX^TX+bI
            cur_mean = a*Inverse(lamda).dot(Transpose(X).dot(new_y))  # μ = aΛ^-1X^TY

        else:
            C = a*Transpose(X).dot(X)+ lamda  # C = aX^TX+Λ
            cur_mean = Inverse(C).dot(a*Transpose(X).dot(new_y) + lamda.dot(pre_mean)) # μ = C^-1(aX^TY+Λμ)
            lamda = C

        mean_pred = (X.dot(cur_mean))[0][0]
        #cur_var_pred = (1/a) + (X.dot(lamda_in).dot(Transpose(X)))[0][0]
        cur_var_pred = a + (X.dot(Inverse(lamda)).dot(Transpose(X)))[0][0]

        print('Posterior mean:')
        PrintMatrix(cur_mean)

        print('\nPosterior variance:')
        PrintMatrix(Inverse(lamda))

        print(f'\nPredictive distribution ~ N({round(mean_pred,5)}, {round(cur_var_pred,5)})')
        print('--------------------------------------------------')

        if abs(pre_var_pred - cur_var_pred) < 0.0001 and num >= 1000:
            break

        pre_mean = cur_mean
        pre_var_pred = cur_var_pred


        if num == 10:
                m_10 = cur_mean.copy()
                x_10 = x.copy()
                y_10 = y.copy()
                a_10 = a
                lambda_inverse_10 = Inverse(lamda).copy()
        elif num == 50:
                m_50 = cur_mean.copy()
                x_50 = x.copy()
                y_50 = y.copy()
                a_50 = a
                lambda_inverse_50 = Inverse(lamda).copy()

    print(time.time() - START)
    SubplotResult(221, 'Ground truth',     None, None, w,                       None, None,                    a_err, True)
    SubplotResult(222, 'Predict result',   x,    y,    np.reshape(cur_mean, n), a   ,Inverse(lamda),          None,  False)
    SubplotResult(223, 'After 10 incomes', x_10, y_10, np.reshape(m_10, n),     a_50, lambda_inverse_10,       None,  False)
    SubplotResult(224, 'After 50 incomes', x_50, y_50, np.reshape(m_50, n),     a_50, lambda_inverse_50,       None,  False)
    plt.tight_layout()
    plt.show()


# In[ ]:




