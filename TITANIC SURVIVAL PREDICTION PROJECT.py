#!/usr/bin/env python
# coding: utf-8

# ## TITANIC SURVIVAL PREDICTION PROJECT

# In[1]:


### First we import the library such as numpy,pandas,matplotlib,seaborn.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


## Now Import the data set with the help of pandas
titanic_data = pd.read_csv("C:\\Users\\Rohit Sharma\\AppData\\Local\\Temp\\Temp1_archive.zip\\tested.csv")


# In[4]:


titanic_data


# In[5]:


## To see the Top 5 rows from the data set
titanic_data.head()


# In[7]:


## To see the last 5 rows from the data set
titanic_data.tail()


# In[8]:


### To get the  information about the data set like datatype,non-null count
titanic_data.info()


# In[ ]:





# In[9]:


## To check the unique value in the data 
titanic_data.nunique()


# In[10]:


## Lets first check the Null value in the data set
titanic_data.isnull()


# In[11]:


## Lets count the null value in the column of the given data set
titanic_data.isnull().sum()


# In[13]:


## calculate the percentage of missing values in each column
(titanic_data.isnull().sum()/(len(titanic_data)))*100


# In[14]:


## visualize missing data using Seaborn’s heatmap.
sns.heatmap(titanic_data.isnull(),yticklabels = False,cbar=False,cmap = 'viridis')
plt.title("Missing Values",fontsize = 12)
plt.xlabel("Columns",fontsize=12)
plt.ylabel("Missing Values",fontsize = 10)
plt.show()



# In[ ]:





# In[15]:


sns.set_style('whitegrid')
sns.countplot(x ="Survived",hue ="Sex",data = titanic_data,palette = 'rainbow')


# In[16]:


sns.set_style("whitegrid")
sns.countplot(x='Survived',hue='Pclass',data = titanic_data, palette = 'rainbow')


# ## Handling the missing value

# In[18]:


# Drop the cabin column from the data set
titanic_data = titanic_data.drop(columns = 'Cabin',axis=1)
titanic_data


# In[19]:


## replacing the missinng value in the age columns with the mean value
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace = True)


# In[20]:


titanic_data["Fare"].fillna(titanic_data["Fare"].mean(),inplace = True)


# In[21]:


titanic_data.isnull().sum()


# ## Data Analysis

# In[22]:


## lets see the stastics value of the given column 
titanic_data.describe()


# In[23]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# In[24]:


titanic_data['Sex'].value_counts()


# ## Data Visualization

# In[25]:


## Here we are checking how many are survived or not.
sns.set_style("whitegrid")
sns.countplot(x='Survived',data = titanic_data)


# In[26]:


## making a count plot for "sex" column 
sns.set_style("whitegrid")
sns.countplot(x='Sex',data = titanic_data)


# In[27]:


## Number of surviver genderwise
sns.countplot(x='Survived', hue='Sex', data = titanic_data)


# In[28]:


## making a count plot for Pclass
sns.countplot(x="Pclass",data = titanic_data)


# In[29]:


sns.countplot(x="Pclass",hue = "Survived",data = titanic_data)


# ## Encoding the Categorical value

# In[30]:


titanic_data["Sex"].value_counts()


# In[31]:


titanic_data["Embarked"].value_counts()


# In[32]:


## coverting categorical columns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{"S":0,"C":1,"Q":2}},inplace = True)


# In[33]:


titanic_data.head()


# ## Seperate features and target from the data set/Data Features

# In[34]:


x= titanic_data.drop(columns =['PassengerId','Name','Ticket','Survived'],axis=1)
y=titanic_data["Survived"]


# In[35]:


print(x)


# In[36]:


print(y)


# ## Splitting the traning data and testing data

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[38]:


# create the 4 array
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)


# In[39]:


print(x.shape,x_train.shape,x_test.shape)


# ## Model Training
# 
# # Logistic Regression

# In[40]:


model = LogisticRegression()


# In[41]:


# training the logistic Regression Model with training data
model.fit(x_train,y_train)


# ## Model Evaluation
# 
# ### accuracy score

# In[42]:


## accuracy on traning data
x_train_prediction = model.predict(x_train)


# In[43]:


print(x_train_prediction)


# In[44]:


traning_data_accuracy = accuracy_score(y_train,x_train_prediction)
print('Accuracy Of traing data is:',traning_data_accuracy)


# In[45]:


## accuracy on testing data
x_test_prediction = model.predict(x_test)


# In[46]:


print(x_test_prediction)


# In[47]:


testing_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Accuracy Of testing data is:',testing_data_accuracy)


# In[ ]:




