#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import binascii,math


# In[2]:


def Load():
    train_image_file=open('train-images.idx3-ubyte','rb')
    train_label_file=open('train-labels.idx1-ubyte','rb')
    test_image_file=open('t10k-images.idx3-ubyte','rb')
    test_label_file=open('t10k-labels.idx1-ubyte','rb')
    
    #Train_image Train_label
    train_image_file.read(4)   #magic
    train_label_file.read(4)   #magic
    train_size = int(binascii.b2a_hex(train_image_file.read(4)),16)
    image_row = int(binascii.b2a_hex(train_image_file.read(4)),16)
    image_col = int(binascii.b2a_hex(train_image_file.read(4)),16)
    train_label_file.read(4)
    
    
    #Test_image Test_label
    test_image_file.read(4)   #magic
    test_label_file.read(4)   #magic
    test_size = int(binascii.b2a_hex(test_image_file.read(4)), 16)
    test_image_file.read(4)    #same as train
    test_image_file.read(4)
    test_label_file.read(4)
    
    return train_image_file,train_label_file,train_size,image_row,image_col,test_image_file,test_label_file,test_size
    


# In[3]:


def Draw(image, image_row, image_col, mode):
    if mode==0:
        for i in range(10):
            print(f'{i}:')
            for j in range(image_row):
                for k in range(image_col):
                    white = sum(image[i][j * image_row + k][:17])
                    black = sum(image[i][j * image_row + k][17:])
                    print(1 if black > white else 0, end=' ')
                print('')
            print('')
    if mode==1:
        print('Imagination of numbers in Bayesian classifier:')
        for i in range(10):
            print(i,":")
            for j in range(image_row):
                for k in range(image_col):
                    print(1 if image[i][j*image_row + k] > 128 else 0,end=' ')
                print('')
            print('')
        
    


# In[4]:


def PrintPostirior(prob,answer):
    print('Posterior (in log scale):')
    for i in range(prob.shape[0]):
        print(f'{i}: {prob[i]}')
    pred = np.argmin(prob)
    print(f'Prediction: {pred}, Ans: {answer}\n')
    return 0 if answer == pred else 1


# In[5]:


def Discrete():
    train_image_file,train_label_file,train_size,image_row,image_col,test_image_file,test_label_file,test_size = Load()
    image_size = image_row * image_col
    image = np.zeros((10, image_size, 32))
    image_sum = np.zeros((10, image_size))
    number = np.zeros((10))
    error = 0
    
    for i in range(train_size):
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        number[label] += 1
        for j in range(image_size):
            gray = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            image[label][j][gray// 8] += 1
            image_sum[label][j] += 1


    for i in range(test_size):
        # print(i, error)
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10))
        test_image = np.zeros((image_size))
        
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)
        for j in range(10):
            # consider number
            prob[j] += np.log(number[j] / train_size)
            for k in range(image_size):
                likelihood = image[j][k][int(test_image[k]/8)]
                if likelihood == 0:
                    likelihood = np.min(image[j][k][np.nonzero(image[j][k])])
                # likelihood = 0.000001 if likelihood == 0 else likelihood
                prob[j] += np.log(likelihood / image_sum[j][k])
        # normalize
        summation = sum(prob)
        prob /= summation
        error += PrintPostirior(prob, answer)

    Draw(image, image_row, image_col, 0)
    print(f'Error rate: {error / test_size}')


# In[6]:


def Continuous():
    train_image_file,train_label_file,train_size,image_row,image_col,test_image_file,test_label_file,test_size = Load()
    
    image_size = image_row * image_col    
    number = np.zeros((10))
    var = np.zeros((10, image_size))
    mean = np.zeros((10, image_size))
    mean_square = np.zeros((10, image_size))

    for i in range(train_size):
        label = int(binascii.b2a_hex(train_label_file.read(1)), 16)
        number[label] += 1
        for j in range(image_size):
            gray = int(binascii.b2a_hex(train_image_file.read(1)), 16)
            mean[label][j] += gray
            mean_square[label][j] += (gray ** 2)
    
    for i in range(10):
        for j in range(image_size):
            mean[i][j] /= number[i]
            mean_square[i][j] /= number[i]
            var[i][j] = mean_square[i][j] - (mean[i][j] ** 2)
            var[i][j] = 10 if var[i][j] == 0 else var[i][j]

    # testing
    error = 0
    for i in range(test_size):
        # print(i, error)
        answer = int(binascii.b2a_hex(test_label_file.read(1)), 16)
        prob = np.zeros((10), dtype=np.float)
        test_image = np.zeros((image_size), dtype=np.int32)
        for j in range(image_size):
            test_image[j] = int(binascii.b2a_hex(test_image_file.read(1)), 16)
        for j in range(10):
            # consider number
            prob[j] += np.log(number[j] / train_size)
            for k in range(image_size):
                # consider likelihood
                likelihood = -0.5 * (np.log(2 * math.pi * var[j][k]) + ((test_image[k] - mean[j][k]) ** 2) / var[j][k])
                prob[j] += likelihood
        # normalize
        summation = sum(prob)
        prob /= summation
        error += PrintPostirior(prob, answer)

    Draw(mean, image_row, image_col, 1)
    print(f'Error rate: {error / test_size}')


# In[ ]:


mode = input("0: discrete mode , 1: continuous mode")
if mode == '1':
    Continuous()
else:
    Discrete()


# In[ ]:




