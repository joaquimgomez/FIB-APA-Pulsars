####
#	Autores: Ferran Velasco y Joaquin Gomez
####

# coding: utf-8

# # Data Exploration & Preprocessing

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Data

# In[2]:


df = pd.read_csv("./data/HTRU2/HTRU_2.csv", names = ['Profile_mean', 'Profile_stdev', 'Profile_skewness',
                                                      'Profile_kurtosis', 'DM_mean', 'DM_stdev', 'DM_skewness',
                                                      'DM_kurtosis', 'class'])

df


# ## Exploration

# ### Lost values

# In[3]:


df.isnull().values.any() # Has no NaN/lost values


# In[4]:


df.isna().values.any()


# ### Statistic analysis

# Data boxplot:

# In[5]:


sns.boxplot(data = df, palette="colorblind")


# Description of the features:

# In[6]:


df.describe()


# In[7]:


pd.set_option('display.max_columns', 500)
df.groupby('class').describe()
#df.describe()


# Plotting 2 to 2 and densities:

# In[8]:


col = df['class'].map({1:'r', 0:'b'})
pd.plotting.scatter_matrix(df, c=col, figsize=(15,15))


# Densities per class and feature:

# In[11]:


dfPulsar = df.loc[df['class'] == 1]
dfNotPulsar = df.loc[df['class'] == 0]

for column in df.columns[:-1]:
    pulsars = dfPulsar[column]
    notPulsars = dfNotPulsar[column]

    p1=sns.kdeplot(pulsars, shade=True, color="r", label="pulsar")
    p1=sns.kdeplot(notPulsars, shade=True, color="b", label="not pulsar")
    plt.title(column)
    plt.show()


# ## Preprocessing

# ### Standarization

# In[8]:


scaler = StandardScaler()

scaledData = scaler.fit_transform(df.drop(['class'], axis = 1))

stdDf = pd.DataFrame(scaledData, columns = df.columns[:-1])
stdDfWithClass = pd.concat([stdDf, df[['class']]], axis = 1)


# In[9]:


stdDfWithClass.to_csv("./data/stdHTRU_2.csv", index = False)


# ### Feature Extraction

# Correlation Matrix of Data:

# In[10]:


corrStd = stdDf.corr()
corrStd.style.background_gradient(cmap='coolwarm')


# In order to improve the performance of ML models that will be affected by the correlation of features and irrelevant variables, we will remove the correlated features (with correlation higher than 0.9).

# In[11]:


features = np.full((corrStd.shape[0],), True, dtype=bool)
for i in range(corrStd.shape[0]):
    for j in range(i+1, corrStd.shape[0]):
        if corrStd.iloc[i,j] >= 0.9:
            if features[j]:
                features[j] = False

selectedFeatures = stdDf.columns[features]

noCorrStdData = stdDf[selectedFeatures]


# In[12]:


noCorrStdDfWithClassData = pd.concat([noCorrStdData, df[['class']]], axis = 1)


# In[13]:


corrNoCorrStd = noCorrStdDfWithClassData.corr()
corrNoCorrStd.style.background_gradient(cmap='coolwarm')


# In[14]:


noCorrStdDfWithClassData.to_csv("./data/noCorrStdHTRU_2.csv", index = False)
