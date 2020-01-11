
# coding: utf-8

# # Linear Model 2 - Logistic regression

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from statsmodels.genmod.families.family import Binomial
from statsmodels.tools.tools import add_constant
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from auxiliars import *
import pickle


# In[3]:


np.random.seed(543)


# ## Data

# Standarized data loading:

# In[4]:


data = pd.read_csv("./data/stdHTRU_2.csv")


# We split a separate test set of relative size 20%:

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(data[data.columns[0:8]], 
                                                    data['class'], 
                                                    test_size = 0.2,
                                                    random_state = 1234)
print(data[data.columns[0:8]])


# I order to improve the performance of logistic regression, we will also analyze the performance of the method with no-correlated standarized data: 

# In[8]:


noCorrData = pd.read_csv("./data/noCorrStdHTRU_2.csv")


# In[9]:


X_train_NC, X_test_NC, y_train_NC, y_test_NC = train_test_split(noCorrData[noCorrData.columns[0:6]], 
                                                    noCorrData['class'], 
                                                    test_size = 0.2)

print(noCorrData[noCorrData.columns[0:6]])


# ## Model Training

# Scikit-learn library offersa method for Logistic Regression classification.

# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


LR = LogisticRegression(n_jobs = -1)


# LogisticRegression allow us to hypertuning the following parameters:
# - Penalty: Used to specify the norm used in the penalization.
#     - L1: Lasso regression.
#     - L2: Ridge regression.
# - C: Inverse of regularization strength
# - Algorithm to use in the optimization problem:
#     - liblinear: for small datasets.
#     - saga: for larger datasets.

# In order to hypertuning model parameters and get a better idea on how the model performs on unseen data, we will use GridSearchCV.

# In[12]:


from sklearn.model_selection import GridSearchCV


# Values of the 10-Fold CV Grid to test:

# In[13]:


grid = {'penalty' : ['l1','l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['liblinear', 'saga']}


# In[14]:


grid


# Grid Search 10-Fold CV:

# In[15]:


gs10cv = GridSearchCV(LR, param_grid = grid, cv = 10, n_jobs = -1)


# ### Normal Data Training

# In[16]:


gs10cv.fit(X_train, y_train)


# In[17]:


gs10cv.best_params_


# In[18]:


pd.DataFrame(gs10cv.cv_results_).iloc[gs10cv.best_index_]


# In[19]:


# Save model
LRFile = open('./models/LR_BestCV_STDData_pickle_file', 'wb')
pickle.dump(gs10cv, LRFile) 


# ### No-correlated Data Training

# Grid Search 10-Fold CV:

# In[20]:


gs10cv_nc = GridSearchCV(LR, param_grid = grid, cv = 10, n_jobs = -1)


# Training:

# In[21]:


gs10cv_nc.fit(X_train_NC, y_train_NC)


# In[22]:


pd.DataFrame(gs10cv_nc.cv_results_).iloc[gs10cv_nc.best_index_]


# In[23]:


# Save model
LRFileNC = open('./models/LR_BestCV_NCorrSTDData_pickle_file', 'wb')
pickle.dump(gs10cv_nc, LRFile)


# ## Testing 

# ### Normal Data Model Testing

# In[48]:


y_pred = gs10cv.predict(X_test)
print("Confusion Matrix:")
confusionMatrix(y_test, y_pred, classes = [0,1])


# In[47]:


print(classification_report(y_test, y_pred, target_names=['no', 'yes']))


# In[44]:


print("Test Error:")
(1-accuracy_score(y_test, y_pred))*100


# ### No-correlated Data Model Testing

# In[26]:


y_pred_NC = gs10cv_nc.predict(X_test_NC)
print("Confusion Matrix:")
confusionMatrix(y_test_NC, y_pred_NC, classes = [0,1])


# In[50]:


print(classification_report(y_test_NC, y_pred_NC, target_names=['no', 'yes']))


# In[27]:


print("Test Error:")
(1-accuracy_score(y_test_NC, gs10cv_nc.predict(X_test_NC)))*100

