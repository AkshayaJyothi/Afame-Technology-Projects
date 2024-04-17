#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Peroject 

# IMPORTING LIBRARIES

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# IMPORTING DATASETS 

# In[3]:


titanic = pd.read_csv('downloads/Titanic-Dataset.csv')


# In[4]:


titanic


# In[5]:


titanic.shape


# VISUALIZATION

# In[6]:


import seaborn as sns
sns.heatmap(titanic.corr(numeric_only=True),cmap="YlGnBu")
plt.show


# In[7]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split (titanic, titanic[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic.loc[train_indices]
    strat_test_set = titanic.loc[test_indices]


# In[8]:


plt.subplot(1,2, 1)
strat_train_set['Survived'].hist ()
strat_train_set['Pclass'].hist ()

plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

plt.show()


# In[9]:


strat_train_set.info()


# Scaling the data

# In[10]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X,Y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X["Age"] = imputer.fit_transform(X[["Age"]])
        return X


# In[11]:


from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator,TransformerMixin):

      def fit(self, X, Y=None):
          return self

      def transform(self, X):
          encoder = OneHotEncoder()
          matrix = encoder.fit_transform(X[["Embarked"]]).toarray()

          column_names = ["C", "S", "Q", "N"]

          for i in range(len(matrix.T)):
             X[column_names [1]] = matrix. T[i]
             
          matrix = encoder.fit_transform (X[ ['Sex']]) .toarray ()

          column_names = ["Female", "Male"]

          for i in range (len (matrix.T)) :
             X [column_names [i]] = matrix.T[i]
             
          return X


# In[12]:


class FeatureDropper (BaseEstimator, TransformerMixin) :
    
    def fit (self, X, Y=None) :
        return self
        
    def transform (self, X) :
        
        return X. drop (["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")


# In[13]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([("ageimputer", AgeImputer()),
                     ("featureencoder", FeatureEncoder()) ,
                     ("featuredropper", FeatureDropper())])


# Spliting the data into traning and testing data

# In[14]:


strat_train_set = pipeline.fit_transform(strat_train_set)


# In[15]:


strat_train_set


# In[16]:


strat_train_set.info()


# In[17]:


from sklearn.preprocessing import StandardScaler
X = strat_train_set.drop (['Survived'], axis=1)
y = strat_train_set ['Survived']

scaler = StandardScaler ()
X_data = scaler. fit_transform (X)
Y_data = y. to_numpy ()


# ML model

# In[18]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier ()

param_gird = [
     {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2,3,4]}
]

grid_search = GridSearchCV(clf, param_gird, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit (X_data, Y_data)


# In[19]:


final_clf = grid_search.best_estimator_


# In[20]:


final_clf


# In[21]:


RandomForestClassifier(max_depth=5, n_estimators=500)


# In[22]:


strat_test_set = pipeline.fit_transform(strat_test_set)


# In[24]:


X_test = strat_test_set.drop(['Survived'], axis=1)
Y_test = strat_test_set['Survived']

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
Y_data_test = Y_test.to_numpy()


# In[25]:


final_clf.score(X_data_test, Y_data_test)


# In[26]:


final_data = pipeline.fit_transform(titanic)


# In[27]:


final_data


# In[28]:


X_final = final_data.drop(['Survived'], axis=1)
Y_final = final_data['Survived']

scaler = StandardScaler()
X_data_final = scaler.fit_transform(X_final)
Y_data_final = Y_final.to_numpy()


# In[29]:


prod_clf = RandomForestClassifier ()

param_gird = [
     {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2,3,4]}
]

grid_search = GridSearchCV(prod_clf, param_gird, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data_final, Y_data_final)


# In[31]:


prod_final_clf = grid_search.best_estimator_


# In[32]:


prod_final_clf

