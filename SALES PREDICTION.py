#!/usr/bin/env python
# coding: utf-8

# ## SALES PREDICTION PROJECT

# In[37]:


## import the libraby for the data analysis, wrangling and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:


## Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[39]:


## import the data set 
df_sales = pd.read_csv("C:\\Users\\Rohit Sharma\\Downloads\\archive (2)\\advertising.csv")


# In[40]:


df_sales


# In[42]:


df_sales.head()


# In[43]:


# to get the information about the data set
df_sales.info()


# In[45]:


# to check the unique value in the data set
df_sales.nunique()


# In[46]:


# lets check the null value in the data set
df_sales.isnull()


# In[47]:


df_sales.isnull().sum()


# In[48]:


df_sales.describe()


# In[52]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df_sales['TV'], ax = axs[0])
plt2 = sns.boxplot(df_sales['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df_sales['Radio'], ax = axs[2])
plt.tight_layout()


# In[54]:


# Exploratory Data Analysis
sns.boxplot(df_sales['Sales'])
plt.show()


# In[55]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(df_sales, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[56]:


# Let's see the correlation between different variables.
sns.heatmap(df_sales.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[74]:


##Seperate features and target from the data set/Data Features
X= df_sales["TV"]
y=df_sales["Sales"]


# ## Model Tranning
# 
# ### Linear Regression

# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[76]:


# Let's now take a look at the train dataset

X_train.head()


# In[77]:


y_train.head()


# In[78]:


import statsmodels.api as sm


# In[79]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[80]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[81]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# In[83]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.94 + 0.054*X_train, 'r')
plt.show()


# ## Model Evaluation

# In[84]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[85]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[86]:


plt.scatter(X_train,res)
plt.show()


# In[87]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[88]:


y_pred.head()


# In[89]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[90]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# In[92]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




