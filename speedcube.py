#!/usr/bin/env python
# coding: utf-8

# #### import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# In[189]:


data = pd.read_json('data/times.json')['session1'].values


# In[190]:


# get only the times
times = []
i = 1
for l in data:
    nl = list()
    nl.append(i)
    nl.append(l[0][1]/1000)
    times.append(nl)
    i += 1
times = np.array(times)


# In[191]:


# Calculate the 10th and 90th percentile thresholds to filter the times
lower_threshold = np.percentile(times[:,1], 10)
upper_threshold = np.percentile(times[:,1], 90)

filtered_times = times[(times[:,1] > lower_threshold) & (times[:,1] < upper_threshold)]
lower_threshold, upper_threshold


# In[192]:


y = times[:, 1]
X = times[:, 0]

from sklearn import linear_model
lin1 = linear_model.LinearRegression()
Xsample = np.c_[X]
ysample = np.c_[y]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
t0, t1


# In[193]:


# plot the graph using matplotlib
plt.scatter(X, y, s=5)
plt.xlabel('Index')
plt.ylabel('Time')
plt.title('Time vs Index')
plt.plot(x, t0 + t1*x, "b")
plt.show()


# In[194]:


# Machine learning to predict my average in certain number of solves
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
model.fit(Xsample, ysample)
X_new = [[1000]]
print(model.predict(X_new))


# In[195]:


y = filtered_times[:, 1]
X = filtered_times[:, 0]

from sklearn import linear_model
lin1 = linear_model.LinearRegression()
Xsample = np.c_[X]
ysample = np.c_[y]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
t0, t1

# plot the graph using matplotlib
plt.scatter(X, y, s=5)
plt.xlabel('Index')
plt.ylabel('Filtered Time')
plt.title('Filtered Time vs Index')
plt.plot(x, t0 + t1*x, "b")
plt.show()


# In[197]:


# Machine learning to predict my average in certain number of solves
import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
model.fit(Xsample, ysample)
X_new = [[200]]
print(model.predict(X_new))

