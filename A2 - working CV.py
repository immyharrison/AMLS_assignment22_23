#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing#
from pathlib import Path
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[22]:


#import conusion matrix (plot)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# In[3]:


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
x_train = get_images('Datasets\\celeba\\img\\')
x_train = np.reshape(x_train, (x_train.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)


# In[4]:


# import y label train
dataFrame = pd.read_csv('Datasets\\celeba\\labels.csv',header = None, prefix="data")
dataFrame['data1']= dataFrame['data0'].str.split('\t')
df3 = pd.DataFrame(dataFrame['data1'].to_list(), columns=['image_number','jpg','gender','smiling'])
df3['smiling'] = df3['smiling'].replace(['-1'], '0')
  
df3 = df3.sort_values(by ='image_number')
df3 = df3.drop(0)
df3['smiling'] = pd.to_numeric(df3['smiling'])
y_train = df3['smiling']


# In[ ]:





# In[ ]:





# In[12]:


#split data 
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2) 


# In[14]:


print(len(x_train),len(y_train))


# # CV to determine the most effective model

# In[29]:


CV_df = pd.DataFrame({"METHOD":[],"MEAN":[],
                         "STD":[]})
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[31]:


# create model
model= LogisticRegression()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_LR = {'Method':'Logistic regression','MEAN':scores.mean(),'STD':scores.std()}


# In[32]:


# support vecotr machine 
model = LinearSVC()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_SV = {'Method':'Support Vector','MEAN':scores.mean(),'STD':scores.std()}


# In[33]:


model = DecisionTreeClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_DT = {'Method':'Decision Tree','MEAN':scores.mean(),'STD':scores.std()}


# In[34]:


model =RandomForestClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_RF = {'Method':'Random Forest','MEAN':scores.mean(),'STD':scores.std()}


# In[35]:


model = KNeighborsClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_KN = {'Method':'K-Nearest Neighbors','MEAN':scores.mean(),'STD':scores.std()}


# In[36]:


list_of_dict = CV_df_LR, CV_df_SV, CV_df_DT,CV_df_RF,CV_df_KN
# Create Dataframe from list of dictionaries and
# pass another list as index
df = pd.DataFrame(list_of_dict)
df


# In[11]:





# # Random forest classifier 

# In[44]:


#load classifier 
models =RandomForestClassifier()
    
# Fit the classifier
models.fit(x_train, y_train)
    
# Make predictions
predictions = models.predict(x_test)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test)
print(accuracy)


# In[55]:


# explore random forest number of trees effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
 
X = x_train
y = y_train
 
# get a list of models to evaluate
def get_models():
 models = dict()
 # define number of trees to consider
 n_trees = [10, 50, 100, 500 ,1000]
 for n in n_trees:
     models[str(n)] = RandomForestClassifier(n_estimators=n)
 return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
 # define the evaluation procedure
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # evaluate the model and collect the results
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
 return scores
 

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 # evaluate the model
 scores = evaluate_model(model, X, y)
 # store the results
 results.append(scores)
 names.append(name)
 # summarize the performance along the way
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# In[ ]:





# In[5]:


# explore random forest number of trees effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
 
X = x_train
y = y_train
 
# get a list of models to evaluate uuu
def get_models():
 models = dict()
 # define number of trees to consider
 max_depth =[10,20,30,40,50,60,70,80,90,None]
 for n in max_depth:
     models[str(n)] = RandomForestClassifier(max_depth=n)
 return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
 # define the evaluation procedure
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # evaluate the model and collect the results
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
 return scores
 

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
 # evaluate the model
 scores = evaluate_model(model, X, y)
 # store the results
 results.append(scores)
 names.append(name)
 # summarize the performance along the way
 print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# In[68]:


len(y_train)


# In[8]:



# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110, None],
    'max_features': [2, 3, 'auto'],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 5, 8, 10, 12],
    'n_estimators': [10, 50, 100, 200, 300,]}

# Create a based model
rf =RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[9]:


# Fit the grid search to the data
grid_search.fit(x_train, y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_


# In[10]:


best_grid


# In[ ]:





# In[ ]:





# In[15]:


#load classifier 
models =RandomForestClassifier()
    
# Fit the classifier
models.fit(x_train, y_train)
    
# Make predictions
predictions = models.predict(x_test)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test)
print(accuracy)


# In[14]:


#load classifier 
models =RandomForestClassifier(max_depth=90, min_samples_leaf=5, min_samples_split=8,
                       n_estimators=300)
    
# Fit the classifier
models.fit(x_train, y_train)
    
# Make predictions
predictions = models.predict(x_test)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test)
print(accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# # Test 

# In[34]:


# y test
test_dataFrame = pd.read_csv('Datasets\\celeba_test\\labels.csv',header = None, prefix="data")
test_dataFrame['data1']= test_dataFrame['data0'].str.split('\t')
df3 = pd.DataFrame(test_dataFrame['data1'].to_list(), columns=['image_number','jpg','gender','smiling'])
df3['smiling'] = df3['smiling'].replace(['-1'], '0')
  
df3 = df3.sort_values(by ='image_number')
df3 = df3.drop(0)
df3['smiling'] = pd.to_numeric(df3['smiling'])
y_test_test_data = df3['smiling']


# In[ ]:





# In[36]:


# load x test data
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
x_test_test_data = get_images('Datasets\\celeba_test\\img\\')
x_test_test_data = np.reshape(x_test_test_data, (x_test_test_data.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_test_test_data)
x_test_test_data = scaler.transform(x_test_test_data)


# In[37]:


# Make predictions
predictions = models.predict(x_test_test_data)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test_test_data)
print(accuracy)


# In[ ]:





# In[39]:


# plot confusion matrix 
plot_confusion_matrix(models, x_test_test_data, y_test_test_data, cmap=plt.cm.Blues)  
plt.show()


# In[ ]:




