#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[22]:


df = pd.read_csv('Iris.csv')


# In[23]:


df = df.iloc[:,1:]


# In[24]:


df['Species'] = df['Species'].str.split("-").str.get(1)


# In[26]:


df.describe()


# In[28]:


df.info()


# In[29]:


df


# In[30]:


df['Species'].unique()


# In[33]:


lb = LabelEncoder()

df['Species'] = lb.fit_transform(df['Species'])


# In[34]:


df['Species'].unique()


# In[35]:


df


# In[38]:


sns.pairplot(df)


# In[39]:


sns.heatmap(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']].corr(), annot=True)


# In[41]:


x=df.drop("Species",axis=1)
y=df['Species']


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.2 , random_state=42)


# In[44]:


from sklearn.tree import DecisionTreeClassifier


# In[45]:


dct =DecisionTreeClassifier()


# In[46]:


dct.fit(x_train,y_train)


# In[47]:


from sklearn import tree


# In[52]:


plt.figure(figsize=(16,10))
tree.plot_tree(dct,filled=True)
plt.show()


# In[53]:


y_pred = dct.predict(x_test)


# In[54]:


y_pred


# In[55]:


from sklearn.metrics import accuracy_score, classification_report


# In[56]:


accuracy_score(y_pred, y_test)


# In[57]:


print(classification_report(y_pred,y_test))


# In[ ]:




