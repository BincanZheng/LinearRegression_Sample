#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# # Plot set up

# In[2]:


plt.rcParams['font.sans-serif'] = ['SimHei'] # display Chinese
plt.rcParams['axes.unicode_minus'] = False   # display minus sign
plt.rcParams['figure.dpi'] = 540             # picture dpi
size = 11                                    # font size in picture


# # Sample set up

# In[3]:


N=500            # assuming population size is 500
n=300            # assuming sample size is 300
ylabel = '风速'  # assuming wind speed data
unit = 'm/s'     # unit of wind speed


# # Random Sample Data

# In[4]:


# Sample Data
# make it like real!
data1 = np.random.rand(n)
data2 = np.random.rand(n)

data1 = (data1-0.5)*1.5
data2 = (data2-0.5)*1.5

data1 = np.abs(data1 + [i/100 for i in range(n)])
data2 = np.abs(data2 + [i/100 for i in range(n)])


# # Data Analysis

# In[ ]:


r = np.corrcoef(data1,data2)[0,1]   # correlation
X=data1.reshape(n,1)        # data transfermation
y=data2.reshape(n,1)        # data transfermation
m = np.mean(y-X)            # difference mean
s = np.std(y-X)             # difference standard deviation


# # Plot

# In[6]:


model = LinearRegression()
model.fit(X,y)                   # fit model
intercept = model.intercept_[0]  # get intercept
coef = model.coef_[0][0]         # get coef

X2 = [[np.min(X)],[np.max(X)]]   # take out minimum and maximum data
y2 = model.predict(X2)           # calculate predicted y by minimum and maximum data
X3 = [np.min(X),np.max(X)]       # take out minimum and maximum data for baseline
y3 = [np.min(X),np.max(X)]       # take out minimum and maximum data for baseline

fig2 = plt.figure(figsize=(8,8))
plt.plot(X,y,'k.')     # original scatter plot
plt.plot(X2,y2,'r-')   # model
plt.plot(X3,y3,'b--')  # baseline

plt.legend([ylabel,'y='+str(round(intercept,4))+'+'+str(round(coef,4))+'x','y=x'],
           bbox_to_anchor=(1,1),fontsize = size) # label,model,baseline
plt.text(0,np.max(X),
         'N={}\nn={}\nx={}{}\ns={}{}\nr={}'.format(N,n,round(m,3),unit,round(s,3),unit,round(r,3)),
         verticalalignment='top',fontsize = size,color='r') # size, sample size, difference mean, difference std, correlation
plt.xlabel('实际测得风速'+ylabel,fontdict={'size':size})
plt.ylabel('真实风速'+ylabel,fontdict={'size':size})
plt.title(ylabel+'相关对比图',fontdict={'size':size})
plt.tick_params(labelsize=size-3)
fig2.tight_layout()
plt.show


# In[7]:


# fig2.savefig(r'Sample.png')

