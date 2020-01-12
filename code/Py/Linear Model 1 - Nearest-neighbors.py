####
#	Autores: Ferran Velasco y Joaquin Gomez
####

# coding: utf-8

# # Linear Model 1 - Nearest-neighbor

# In[23]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from auxiliars import *
import pickle


# In[2]:


np.random.seed(1234)


# ## Data

# Standarized data loading:

# In[3]:


data = pd.read_csv("./data/stdHTRU_2.csv")


# We split a separate test set of relative size 20%:

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(data[data.columns[0:8]],
                                                    data['class'],
                                                    test_size = 0.2,
                                                    random_state = 1234)


# I order to improve the performance of k-NN, we will analyze the performance of the method with no-correlated standarized data:

# In[5]:


noCorrData = pd.read_csv("./data/noCorrStdHTRU_2.csv")


# In[6]:


X_train_NC, X_test_NC, y_train_NC, y_test_NC = train_test_split(noCorrData[noCorrData.columns[0:6]],
                                                    noCorrData['class'],
                                                    test_size = 0.2,
                                                    random_state = 1234)


# ## Model Training

# Scikit-learn library offers two options of Supervised Nearest Neighbors:
# - KNeighborsClassifier: Algorithm based on the k number of classes.
# - RadiusNeighborsClassifier: Algorithm based on the number of neighbors within a fixed radius  of each training point.
#
# We will use the first one because we know the number of classes and it is more useful.

# In[7]:


from sklearn.neighbors import KNeighborsClassifier


# In[8]:


kNC = KNeighborsClassifier(n_jobs = -1)


# KNeighborsClassifier allow us to hypertuning the following parameters:
# - Weights:
#     - Uniform: All points in each neighborhood are weighted equally.
#     - Distance: Weight points by the inverse of their distance.
# - Algorithm to compute the nearest neighbors:
#     - BallTree
#     - KDTree
#     - Brute-force Search
# - Power parameter for the Minkowski metric:
#     - Manhattan Distance (p = 1)
#     - Euclidean Distance (p = 2)

# In order to hypertuning model parameters and get a better idea on how the model performs on unseen data, we will use GridSearchCV.

# In[9]:


from sklearn.model_selection import GridSearchCV


# Values of the 10-Fold CV Grid to test:

# In[10]:


grid = {'n_neighbors': np.arange(2, 51),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'p': [1,2]}


# In[11]:


grid


# Grid Search 10-Fold CV:

# In[12]:


gs10cv = GridSearchCV(kNC, param_grid = grid, cv = 10, n_jobs = -1)


# ### Normal Data Training

# In[13]:


gs10cv.fit(X_train, y_train)


# In[14]:


gs10cv.best_params_


# In[15]:


pd.DataFrame(gs10cv.cv_results_).iloc[gs10cv.best_index_]


# In[16]:


# Save model
kNCFile = open('./models/kNC_BestCV_STDData_pickle_file', 'wb')
pickle.dump(gs10cv, kNCFile)


# ### No-correlated Data Training

# Grid Search 10-Fold CV:

# In[17]:


gs10cv_nc = GridSearchCV(kNC, param_grid = grid, cv = 10, n_jobs = -1)


# Training:

# In[18]:


gs10cv_nc.fit(X_train_NC, y_train_NC)


# In[19]:


pd.DataFrame(gs10cv_nc.cv_results_).iloc[gs10cv_nc.best_index_]


# In[20]:


# Save model
kNCFileNC = open('./models/kNC_BestCV_NCorrSTDData_pickle_file', 'wb')
pickle.dump(gs10cv_nc, kNCFile)


# ## Testing

# ### Normal Data Model Testing

# In[21]:


y_pred = gs10cv.predict(X_test)


# In[24]:


print(classification_report(y_test, y_pred))


# In[25]:


print("Confusion Matrix:")
confusionMatrix(y_test, y_pred, classes = [0,1])


# In[26]:


print("Test Error:")
(1-accuracy_score(y_test, gs10cv.predict(X_test)))*100


# ### No-correlated Data Model Testing

# In[27]:


y_pred_NC = gs10cv_nc.predict(X_test_NC)


# In[28]:


print(classification_report(y_test_NC, y_pred_NC))


# In[29]:


print("Confusion Matrix:")
confusionMatrix(y_test_NC, y_pred_NC, classes = [0,1])


# In[30]:


print("Test Error:")
(1-accuracy_score(y_test_NC, gs10cv_nc.predict(X_test_NC)))*100
