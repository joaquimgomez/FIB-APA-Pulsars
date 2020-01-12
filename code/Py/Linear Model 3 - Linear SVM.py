####
#	Autores: Ferran Velasco y Joaquin Gomez
####

# coding: utf-8

# # Linear Model 3 - Linear SVM

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from auxiliars import *
import pickle


# ## Data

# In[2]:


data = pd.read_csv("./data/stdHTRU_2.csv")


# In[3]:


col = data['class'].map({1:'r', 0:'b'})
pd.plotting.scatter_matrix(data.drop(['class'], axis = 1), c=col, figsize=(15,15))


# From the Scatter Matrix we can de deduce that the Linear Kernel should be sufficient for the separation of classes.
#
# Even so, we can obvserve that some features, see for example DM_mean-DM_stdev, have very close data. In order to reduce the impact of this fact, let's train SVM with (standarized) normal data and data with selected features.

# We split a separate test set of relative size 20%:

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(data[data.columns[0:8]],
                                                    data['class'],
                                                    test_size = 0.2,
                                                    random_state = 1234)


# We will analyze the performance of the method with no-correlated standarized data:

# In[5]:


noCorrData = pd.read_csv("./data/noCorrStdHTRU_2.csv")


# In[6]:


X_train_NC, X_test_NC, y_train_NC, y_test_NC = train_test_split(noCorrData[noCorrData.columns[0:6]],
                                                    noCorrData['class'],
                                                    test_size = 0.2,
                                                    random_state = 1234)


# ## Model Training

# In order to train Linear SVM we are going to use the scikit-learn LinearSVC class, specialized in Linear SVM.

# In[7]:


from sklearn.svm import LinearSVC


# In[8]:


SVMClass = LinearSVC(random_state = 1234, max_iter = 5000)


# LinearSVC allow us to hypertuning the following parameters:
# - Regularization parameter C.
# - Class weights:
#     - Dict: Weights specified by class.
#     - Balanced: Uses the values of target (y) to automatically adjust weights inversely proportional to class frequencies in the input data.

# In order to hypertuning model parameters and get a better idea on how the model performs on unseen data, we will use GridSearchCV.

# In[9]:


from sklearn.model_selection import GridSearchCV


# Values of the 10-Fold CV Grid to test:

# In[10]:


grid = {'C': [10**x for x in range(-3, 4, 1)],
        'class_weight': [{0: 1, 1: 1}, 'balanced']}


# In[11]:


grid


# Grid Search 10-Fold CV:

# In[12]:


gs10cv = GridSearchCV(SVMClass, param_grid = grid, cv = 10, n_jobs = -1)


# ### Normal Data Training

# In[13]:


gs10cv.fit(X_train, y_train)


# In[14]:


pd.DataFrame(gs10cv.cv_results_)


# In[15]:


gs10cv.best_params_


# In[16]:


pd.DataFrame(gs10cv.cv_results_).iloc[gs10cv.best_index_]


# In[17]:


# Save model
SVMClassFile = open('./models/SVMClass_BestCV_STDData_pickle_file', 'wb')
pickle.dump(gs10cv, SVMClassFile)


# ### No-correlated Data Training

# Grid Search 10-Fold CV:

# In[18]:


gs10cv_nc = GridSearchCV(SVMClass, param_grid = grid, cv = 10, n_jobs = -1)


# Training:

# In[19]:


gs10cv_nc.fit(X_train_NC, y_train_NC)


# In[20]:


pd.DataFrame(gs10cv_nc.cv_results_)


# In[21]:


gs10cv_nc.best_params_


# In[22]:


pd.DataFrame(gs10cv_nc.cv_results_).iloc[gs10cv_nc.best_index_]


# In[23]:


# Save model
SVMClassFileNC = open('./models/SVMClass_BestCV_NCorrSTDData_pickle_file', 'wb')
pickle.dump(gs10cv_nc, SVMClassFileNC)


# ## Testing

# ### Normal Data Model Testing

# In[24]:


y_pred = gs10cv.predict(X_test)


# In[25]:


print(classification_report(y_test, y_pred))


# In[26]:


print ("Confusion Matrix:")
confusionMatrix(y_test, y_pred, classes = [0,1])


# In[27]:


print("Test Error:")
(1-accuracy_score(y_test, gs10cv.predict(X_test)))*100


# ### No-correlated Data Model Testing

# In[28]:


y_pred_NC = gs10cv_nc.predict(X_test_NC)


# In[29]:


print(classification_report(y_test_NC, y_pred_NC))


# In[30]:


print ("Confusion Matrix:")
confusionMatrix(y_test_NC, y_pred_NC, classes = [0,1])


# In[31]:


print("Test Error:")
(1-accuracy_score(y_test_NC, gs10cv_nc.predict(X_test_NC)))*100
