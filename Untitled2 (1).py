#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# In[63]:


economy = pd.read_csv("Indian_Economy_Sectorwise_Dataset.csv")


# In[64]:


economy


# In[65]:


economy.shape


# In[66]:


economy.info()


# In[67]:


economy.describe()


# In[68]:


economy.isnull().sum()


# In[69]:


x = economy[["Year"]]


# In[70]:


y = economy[["Imports_Crore"]]


# In[71]:


x_train, x_test, y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state =42
)


# In[72]:


x_train.shape


# In[78]:


model = LinearRegression()


# In[74]:


model.fit(x_train, y_train)


# In[75]:


model.predict([[2039]])


# In[83]:


joblib.dump(model,'economy.pkl')


# In[ ]:




