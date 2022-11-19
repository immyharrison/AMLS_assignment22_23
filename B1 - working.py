#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
from pathlib import Path


# In[ ]:





# In[6]:


# load x train data
def get_images(path):
    all_images_as_array = []
    label = []
    
    for filename in os.listdir(path):
        img = Image.open(path+filename)
        #print(filename)
        new_img = img.resize((64, 64))
        np_array = np.asarray(new_img)
        all_images_as_array.append(np_array)

    return np.array(all_images_as_array)
x_train = get_images('Datasets\\cartoon_set\\img\\')
x_train = np.reshape(x_train, (x_train.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)


# In[7]:


# import y label train
dataFrame = pd.read_csv('Datasets\\cartoon_set\\labels.csv',header = None, prefix="data")
dataFrame['data1']= dataFrame['data0'].str.split('\t')
df3 = pd.DataFrame(dataFrame['data1'].to_list(), columns=['image_number','eye color','face shape','file name'])
df3['face shape'] = df3['face shape'].replace(['-1'], '0')
  
df3 = df3.sort_values(by ='image_number')
df3 = df3.drop(0)
df3['face shape'] = pd.to_numeric(df3['face shape'])
y_train = df3['face shape']


# In[8]:


#df3[0:20]


# In[ ]:





# In[9]:


# import y label test
test_dataFrame = pd.read_csv('Datasets\\cartoon_set_test\\labels.csv',header = None, prefix="data")
test_dataFrame['data1']= test_dataFrame['data0'].str.split('\t')
test_df3 = pd.DataFrame(test_dataFrame['data1'].to_list(), columns=['image_number','eye color','face shape','file name'])
test_df3['face shape'] = test_df3['face shape'].replace(['-1'], '0')
test_df3 = test_df3.sort_values(by ='image_number')
test_df3 = test_df3.drop(0)
test_df3['face shape'] = pd.to_numeric(test_df3['face shape'])
y_test = test_df3['face shape']
len(y_test)



# In[10]:


# load x test data 
def get_images(path):
    all_images_as_array = []
    label = []
    
    for filename in os.listdir(path):
        img = Image.open(path+filename)
        new_img = img.resize((64, 64))
        np_array = np.asarray(new_img)
        all_images_as_array.append(np_array)

    return np.array(all_images_as_array)
x_test = get_images('Datasets\\cartoon_set_test\\img\\')
x_test = np.reshape(x_test, (x_test.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)


# In[11]:


print(len(x_train),len(y_train),len(x_test),len(y_test))


# In[ ]:





# In[12]:


models = {}

# Logistic Regression
#from sklearn.linear_model import LogisticRegression
#models['Logistic Regression'] = LogisticRegression(solver='saga', max_iter=100)

# Support Vector Machines
#from sklearn.svm import LinearSVC
#models['Support Vector Machines'] = LinearSVC()

# Decision Trees
#from sklearn.tree import DecisionTreeClassifier
#models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
#from sklearn.neighbors import KNeighborsClassifier
#models['K-Nearest Neighbor'] = KNeighborsClassifier()

#from sklearn.linear_model import SGDClassifier
#models['SDG Classifier']=SGDClassifier(learning_rate = 'optimal',  alpha = 1e-5, penalty='l1', max_iter = 3000,shuffle = True, loss ='perceptron',verbose=True, random_state = 42, n_jobs = -1)


# In[13]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy, precision, recall = {}, {}, {}
for key in models.keys():
    
    # Fit the classifier
    models[key].fit(x_train, y_train)
    
    # Make predictions
    predictions = models[key].predict(x_test)
    probas = models[key].predict_proba(x_test)
    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test, average='micro')
    recall[key] = recall_score(predictions, y_test, average='micro')


# In[14]:


import pandas as pd
df_models = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_models['Accuracy'] = accuracy.values()
df_models['Precision'] = precision.values()
df_models['Recall'] = recall.values()
df_models


# In[ ]:





# In[ ]:




