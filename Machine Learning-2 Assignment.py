
# coding: utf-8

# In[1]:


#Build the linear regression model using scikit learn in boston data to predict 'Price'
#based on other dependent variable.


# In[2]:


# import corresponding libraries: 

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# load boston housing data available in scipy into a dataframe bos

boston = load_boston()


# In[4]:


boston.keys()


# In[5]:


print(boston.DESCR)


# In[6]:


boston.feature_names


# In[7]:


boston.target


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize = (6, 4))
plt.hist(boston.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()


# In[9]:


bos = pd.DataFrame(boston.data, columns=boston.feature_names)
bos.head()


# In[10]:


bos['Price'] = boston.target


# In[11]:


bos.describe()


# In[12]:


# Exploaratory Data Analysis 


# In[14]:


# Print the scatter plot for each feature with respect to price

for index, feature_name in enumerate(boston.feature_names):
    plt.figure(figsize = (4, 3))
    plt.scatter(boston.data[:, index], boston.target)
    plt.ylabel('price', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()


# In[15]:


# Split train-test dataset

# Split the dataset into two: target value and predictor values.
# Let’s call the target value Y and predictor values X.
# Thus,
# Y = Boston Housing Price X = All other features


# In[17]:


X = bos.drop('Price', axis=1)
Y = bos.Price
print("X Shape : ", X.shape)
print("Y Shape : ", Y.shape)


# In[18]:


from sklearn.model_selection import train_test_split

# splitting 66.66% for train data and 33.33% for test data.

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)
print("X_train Shape :", X_train.shape)
print("X_test Shape : ", X_test.shape)
print("Y_train Shape : ", Y_train.shape)
print("Y_test Shape : ", Y_test.shape)


# In[19]:


# By applying Linear Regression Model


# In[20]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()

# To train the model

lm.fit(X_train, Y_train)

# To predict the prices based on the test data

Y_pred = lm.predict(X_test)


# In[21]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.hist(Y_pred)
plt.xlabel('Predicted price ($1000s)')
plt.ylabel('count')
plt.tight_layout()


# In[22]:


# print coefficients (Slope) beta0

print(boston.feature_names,"\n", lm.intercept_)


# In[23]:


# print coefficients (Slope) beta1

print(boston.feature_names,"\n", lm.coef_)


# In[24]:


# How do we interpret the coefficients? 


# In[25]:


# Holding all other variables fixed, 
#considering an area where there is more crime rate (CRIM) the predicted price of the house decrease by 1177.88 $.

# Being an urban area with good pupil-teacher ratio by town is associated with an average 
# increase in price of houses by 6324.51 $


# In[26]:


import seaborn as sns
sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Price vs Predicted prices")


# In[27]:


# Error Rate of the Model - Root Mean square error 


# In[28]:


from sklearn.metrics import mean_squared_error
print("Error Rate of the Regression Model : ", mean_squared_error(Y_pred, Y_test))


# In[29]:


bos.columns


# In[31]:


import statsmodels.formula.api as smf

# only include TV and Radio in the model

lm = smf.ols(formula='Price ~ CRIM + ZN + INDUS + CHAS + RM + AGE + DIS + RAD + TAX + PTRATIO', data=bos).fit()
lm.rsquared


# In[34]:


import statsmodels.formula.api as smf

# only include TV and Radio in the model

lm = smf.ols(formula='Price ~ CRIM + ZN + INDUS + CHAS + RM + AGE + DIS + RAD + TAX + PTRATIO + B', data=bos).fit()
lm.rsquared


# In[35]:


# Confidence Interval associated with the model 


# In[36]:


import statsmodels.formula.api as smf

# only include TV and Radio in the model

lm = smf.ols(formula='Price ~ CRIM + ZN + INDUS + CHAS + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=bos).fit()
lm.rsquared


# In[37]:


lm.conf_int()


# In[38]:


lm.summary()

