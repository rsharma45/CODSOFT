#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[13]:


## load the data set in the data frame
credit_card = pd.read_csv("C://Users//Rohit Sharma//Downloads//archive (4)//creditcard.csv")


# In[14]:


credit_card


# In[15]:


credit_card.head()


# In[16]:


credit_card.tail()


# In[17]:


credit_card.info()


# In[18]:


credit_card.describe()


# In[19]:


## check the missing value
credit_card.isnull().sum()


# In[20]:


## distribution of legite transaction & fraudent transaction
credit_card['Class'].value_counts()


# ### This data is highly unbalanced
#  0-> Legite transaction
# 1-> Fraud Transaction

# In[22]:


## Seperate the data for data analysis
legite = credit_card[credit_card.Class==0]
fraud = credit_card[credit_card.Class==1]


# In[23]:


print(legite.shape)
print(fraud.shape)


# In[25]:


## Statistical measure of the data
legite.Amount.describe()


# In[26]:


fraud.Amount.describe()


# In[27]:


## Compare the mean of both the transaction
credit_card.groupby('Class').mean()


# ### Under Sampling
# ##### Build a sample data set contaning a similar distribution  of normal transaction and fraudent transaction
# #### Number of fraudenet transaction is 492

# In[28]:


legit_sample = legite.sample(492)


# ###Concatenating the data set

# In[30]:


new_dataset = pd.concat([legit_sample,fraud],axis=0)


# In[31]:


new_dataset.head()


# In[32]:


new_dataset.tail()


# In[34]:


new_dataset['Class'].value_counts()


# In[37]:


new_dataset.groupby('Class').mean()


# ### splite the data into features and Target

# In[39]:


x= new_dataset.drop(columns='Class',axis=1)
y= new_dataset['Class']


# In[42]:


print(x)


# In[43]:


print(y)


# ### Splitting the data into training and testing data

# In[50]:


x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[51]:


print(x.shape,x_train.shape,x_test.shape)


# ### Model traning
# #### Logistic Regression

# In[52]:


model= LogisticRegression()


# In[53]:


## Traning the logistic model with traning data
model.fit(x_train,y_train)


# #### ModelEvaluation

# In[58]:


## Accuracy Score
## accuracy on traning data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[60]:


print('Accuracy on traning data :', training_data_accuracy)


# In[61]:


### Acuuracy on test data
x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[62]:


print('Accuracy on testing data :', testing_data_accuracy)


# In[ ]:




