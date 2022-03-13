#!/usr/bin/env python
# coding: utf-8

# In[1]:


def F(n):
    return n * F(n - 1) if n > 1 else 1


# In[2]:


def C(N, m):
    if N - m < m:
        m = N - m
    res = 1
    for i in range(m):
        res *= N
        N -= 1
    res /= F(m)
    return res


# In[3]:


sample=[]
with open("testfile.txt") as f:
        for line in f.readlines():
            sample.append(line.strip())
        a = int(input("a = "))
        b = int(input("b = "))
        for i in range(len(sample)):
            print(f'case {i+1}: {sample[i]}')
            m=0
            N=len(sample[i])
            likelihood = 0
            for i in sample[i]:
                if i=='1':
                    m +=1 
                else:
                    m +=0
            p = m/N
            likelihood = C(N,m)*(p**m)*((1-p)**(N - m))
            print(f'Likelihood: {likelihood}')
            print(f'Beta prior: a = {a} b = {b}')
            
            a = a+m
            b = b+(N-m)
            
            print(f'Beta posterior: a = {a} b = {b}')
            print('\n')


# In[ ]:




