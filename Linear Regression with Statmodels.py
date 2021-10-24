#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels import api as sm


# In[6]:


data = pd.read_csv("student_data.csv")


# In[7]:


data.head()


# In[11]:


data.describe()


# In[13]:


X = data["SAT"]
y = data["GPA"]


# In[16]:


x = sm.add_constant(X)
results = sm.OLS(y,x).fit()


# In[17]:


results.summary()


# In[ ]:




