####
#	Autores: Ferran Velasco y Joaquin Gomez
####

# coding: utf-8

# # Non-linear Model 1 - Multi-layer Perceptron

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


# We split a separate test set of relative size 20%:

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(data[data.columns[0:8]],
                                                    data['class'],
                                                    test_size = 0.2,
                                                    random_state = 1234)


# We will analyze the performance of the method with no-correlated standarized data:

# In[16]:


noCorrData = pd.read_csv("./data/noCorrStdHTRU_2.csv")


# In[17]:


X_train_NC, X_test_NC, y_train_NC, y_test_NC = train_test_split(noCorrData[noCorrData.columns[0:6]],
                                                    noCorrData['class'],
                                                    test_size = 0.2,
                                                    random_state = 1234)


# ## Model Training

# In[4]:


from sklearn.neural_network import MLPClassifier


# In[5]:


MLPC = MLPClassifier(random_state = 1234, solver = 'adam', max_iter=100)


# MLPClassifier allow us to hypertuning the following parameters:
# - Hidden Layer Sizes
# - Activation function
#     - Logistic Sigmoid Function (logistic)
#     - Hyperbolic tan Function (tanh)
#     - Rectified Linear Unit Function (relu)
# - Alpha (L2 Regularization)

# In order to hypertuning model parameters and get a better idea on how the model performs on unseen data, we will use GridSearchCV.

# In[6]:


from sklearn.model_selection import GridSearchCV


# Values of the 10-Fold CV Grid to test:

# In[11]:


grid = {'hidden_layer_sizes': [(20,), (40,), (50,), (70,), (100,), (20,20,20), (50,50,50), (20,50,200), (50,100,50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': 10.0 ** -np.arange(1, 7)}


# In[12]:


grid


# Grid Search 10-Fold CV:

# In[13]:


gs10cv = GridSearchCV(MLPC, param_grid = grid, cv = 10, n_jobs = -1)


# ### Normal Data Training

# In[14]:


gs10cv.fit(X_train, y_train)


# In[15]:


gs10cv.best_params_


# In[18]:


pd.DataFrame(gs10cv.cv_results_).iloc[gs10cv.best_index_]


# In[20]:


# Save model
MLPClassFile = open('./models/MLPClass_BestCV_STDData_pickle_file', 'wb')
pickle.dump(gs10cv, MLPClassFile)


# ### No-correlated Data Training

# In[22]:


gs10cv_nc = GridSearchCV(MLPC, param_grid = grid, cv = 10, n_jobs = -1)


# In[23]:


gs10cv_nc.fit(X_train_NC, y_train_NC)


# In[24]:


gs10cv_nc.best_params_


# In[25]:


pd.DataFrame(gs10cv_nc.cv_results_).iloc[gs10cv_nc.best_index_]


# In[26]:


# Save model
MLPClassFileNC = open('./models/MLPClass_BestCV_NCorrSTDData_pickle_file', 'wb')
pickle.dump(gs10cv_nc, MLPClassFileNC)


# ## Testing

# ### Normal Data Model Testing

# In[27]:


y_pred = gs10cv.predict(X_test)


# In[28]:


print(classification_report(y_test, y_pred))


# In[29]:


print ("Confusion Matrix:")
confusionMatrix(y_test, y_pred, classes = [0,1])


# In[30]:


print("Test Error:")
(1-accuracy_score(y_test, gs10cv.predict(X_test)))*100


# ### No-correlated Data Model Testing

# In[31]:


y_pred_NC = gs10cv_nc.predict(X_test_NC)


# In[32]:


print(classification_report(y_test_NC, y_pred_NC))


# In[33]:


print ("Confusion Matrix:")
confusionMatrix(y_test_NC, y_pred_NC, classes = [0,1])


# In[34]:


print("Test Error:")
(1-accuracy_score(y_test_NC, gs10cv_nc.predict(X_test_NC)))*100
