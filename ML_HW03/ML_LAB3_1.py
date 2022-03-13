#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import math


# In[5]:


def Univariate_gaussian_data_generator(mean,std_dev):
    return mean + std_dev * (sum(np.random.uniform(0, 1, 12)) - 6)


# In[6]:


if __name__ == '__main__':
    m = int(input("m = "))
    s = int(input("s = "))
    std_dev = math.sqrt(s)
    print(f'Data point source function: N({m}, {s})\n')

    data_num = 0
    cur_mean = 0
    pre_mean = 0
    cur_vari = 0
    pre_vari = 0

    while 1:
        new_data = Univariate_gaussian_data_generator(m,std_dev)
        print(f'Add data point: {new_data}\n')
        data_num +=1
        cur_mean = ((data_num - 1)*pre_mean + new_data) / data_num        #Welford's online algorithm
        cur_vari = ((data_num - 1)*pre_vari + (new_data - pre_mean)*(new_data - cur_mean))/ data_num
        print(f'Mean = {cur_mean} Variance = {cur_vari}\n')

        if abs(cur_mean - pre_mean) < 0.00001 and abs(cur_vari - pre_vari) < 0.00001:
            break

        pre_mean = cur_mean
        pre_vari = cur_vari


# In[ ]:




