#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


train= pd.read_csv('C:\\Users\\Well\\Downloads\\train_data.csv')
train


# In[5]:


test= pd.read_csv('C:\\Users\\Well\\Downloads\\test_data.csv')
test


# In[103]:


train.original= train.copy()
test.original= test.copy()


# In[8]:


train.columns


# In[9]:


test.columns


# In[10]:


train.dtypes


# In[11]:


test.dtypes


# In[121]:


train.describe()


# In[122]:


test.describe()


# In[14]:


train.shape, test.shape


# In[18]:


train['Loan_Status'].value_counts()


# In[19]:


train['Gender'].value_counts()


# In[20]:


train['Married'].value_counts()


# In[21]:


train['Dependents'].value_counts()


# In[22]:


train['Education'].value_counts()


# In[23]:


train['Self_Employed'].value_counts()


# In[24]:


train['Loan_Status'].value_counts(normalize= True)


# In[34]:


train['Gender'].value_counts(normalize= True).plot.bar(title = 'Gender')
plt.show()
train['Married'].value_counts(normalize= True).plot.bar(title = 'Married')
plt.show()
train['Self_Employed'].value_counts(normalize= True).plot.bar(title = 'Self_Employed')
plt.show()
train['Credit_History'].value_counts(normalize= True).plot.bar(title = 'Credit_History')
plt.show()


# In[38]:


train['Dependents'].value_counts(normalize= True).plot.bar(title = 'Dependents')
plt.show()
train['Property_Area'].value_counts(normalize= True).plot.bar(title = 'Property_Aera')
plt.show()
train['Education'].value_counts(normalize= True).plot.bar(title = 'Education')
plt.show()


# In[43]:


sns.distplot(train['ApplicantIncome'])


# In[45]:


train['ApplicantIncome'].plot.box() 
plt.show()


# In[49]:


train.boxplot(column='ApplicantIncome', by = 'Education')


# In[50]:


sns.distplot(train['CoapplicantIncome'])


# In[51]:


train['CoapplicantIncome'].plot.box() 
plt.show()


# In[53]:


sns.distplot(train.dropna()['LoanAmount'])


# In[54]:


train['LoanAmount'].plot.box() 
plt.show()


# In[55]:


train.isnull().sum()


# In[56]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[57]:


train['Loan_Amount_Term'].value_counts()


# In[62]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[63]:


train.isnull().sum()


# In[65]:


test.isnull().sum()


# In[66]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[68]:


test.isnull().sum()


# In[74]:


train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20)
plt.show()
test['LoanAmount_log'] = np.log(test['LoanAmount'])
test['LoanAmount_log'].hist(bins=20) 
plt.show()


# In[75]:


train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)


# In[76]:


X = train.drop('Loan_Status',1) 
y = train.Loan_Status


# In[77]:


X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# In[90]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.25)


# In[91]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# In[92]:


model = LogisticRegression() 
model.fit(x_train, y_train)


# In[140]:


pred_cv = model.predict(x_test)


# In[141]:


accuracy_score(y_test,pred_cv)


# In[142]:


pred_test = model.predict(test)


# In[143]:


submission=pd.read_csv("C:\\Users\\Well\\Downloads\\Sample_Submission.csv")
submission


# In[144]:


submission['Loan_Status']=pred_test
submission['Loan_Status']


# In[145]:


submission['Loan_ID']= test_original['Loan_ID']


# In[134]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[135]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[ ]:




