#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries 
import pandas as pd
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
from pathlib import Path
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std


# In[2]:


# import ML methods 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


# In[3]:


#import conusion matrix (plot)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[6]:


# load x train data
left = 175
top = 250
right = 320
bottom = 275

all_images_as_array = []
label = []
remove_array = []
keep_array = []
for filename in os.listdir('Datasets\\cartoon_set\\img\\'):
    img = Image.open('Datasets\\cartoon_set\\img\\'+filename)
    #print(filename)
    new_img = img.crop((left, top, right, bottom))
    average = np.average(new_img)
    number = 0
    if average >= 120:
        np_array = np.asarray(new_img)
        np_array = np_array/255
        all_images_as_array.append(np_array)
        keep_array.append(filename)
    if average <120:
            #print(filename)
        remove_array.append(filename)

x_train = np.array(all_images_as_array)
#print(len(x_train))
x_train = np.reshape(x_train, (x_train.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)


# In[7]:


len(x_train)


# In[ ]:





# In[ ]:





# In[8]:


# import y label train
dataFrame = pd.read_csv('Datasets\\cartoon_set\\labels.csv',header = None, prefix="data")
dataFrame['data1']= dataFrame['data0'].str.split('\t')
df3 = pd.DataFrame(dataFrame['data1'].to_list(), columns=['image_number','eye color','face shape','file name'])
df3['eye color'] = df3['eye color'].replace(['-1'], '0')
  
df3 = df3.sort_values(by ='image_number')
df3 = df3.drop(0)
df3['eye color'] = pd.to_numeric(df3['eye color'])
df3_y = df3[['file name','eye color']]
#y_train = df3['eye color']


# In[9]:


# filter for the same as the non-sunglasses images 
df_y = df3_y[df3_y['file name'].isin(keep_array)]

y_train = df_y['eye color']


# In[18]:


len(y_train)


# In[ ]:





# In[10]:


#split data 
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2) 


# In[17]:





# # Cross validation for model selection 

# In[19]:


CV_df = pd.DataFrame({"METHOD":[],"MEAN":[],
                         "STD":[]})
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[20]:


# create model
model= LogisticRegression()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_LR = {'Method':'Logistic regression','MEAN':scores.mean(),'STD':scores.std()}


# In[21]:


# support vecotr machine 
model = LinearSVC()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_SV = {'Method':'Support Vector','MEAN':scores.mean(),'STD':scores.std()}


# In[22]:


model = DecisionTreeClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_DT = {'Method':'Decision Tree','MEAN':scores.mean(),'STD':scores.std()}


# In[23]:


model =RandomForestClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_RF = {'Method':'Random Forest','MEAN':scores.mean(),'STD':scores.std()}


# In[24]:


model = KNeighborsClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_KN = {'Method':'K-Nearest Neighbors','MEAN':scores.mean(),'STD':scores.std()}


# In[25]:


list_of_dict = CV_df_LR, CV_df_SV, CV_df_DT,CV_df_RF,CV_df_KN
# Create Dataframe from list of dictionaries and
# pass another list as index
df = pd.DataFrame(list_of_dict)
df


# # Hyperparameter tuning

# In[27]:



# Create the parameter grid based on the results of random search 
param_grid = {'C':[100,10,1.0,0.1,0.01]}

# Create a based model
lr = LogisticRegression()
grid_search = GridSearchCV(estimator = lr, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[28]:


# Fit the grid search to the data
grid_search.fit(x_train, y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_


# In[29]:


best_grid


# In[30]:


# result of CV into tabel 
import pandas as pd

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(
    results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
).rename_axis("kernel")
results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]


# In[31]:


# extract paramets and test scores 
r_df = results_df[['param_C','split0_test_score','split1_test_score' , 'split2_test_score']]


# In[32]:


# plot test scores 
df = r_df.set_index('param_C')
df.T.boxplot()


# In[34]:


#load classifier 
models = LogisticRegression(C= 0.01)
    
# Fit the classifier
models.fit(x_train, y_train)
    
# Make predictions
predictions = models.predict(x_test)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test)
print(accuracy)


# # Testing model

# In[49]:


all_images_as_array_test = []
label = []
remove_array_test = []
keep_array_test = []
for filename in os.listdir('Datasets\\cartoon_set_test\\img\\'):
    img = Image.open('Datasets\\cartoon_set_test\\img\\'+filename)
    #print(filename)
    new_img = img.crop((left, top, right, bottom))
    average = np.average(new_img)
    number = 0
    if average >= 120:
        np_array = np.asarray(new_img)
        np_array = np_array/255
        all_images_as_array_test.append(np_array)
        keep_array_test.append(filename)
    if average <120:
            #print(filename)
        remove_array_test.append(filename)

x_test_test_data = np.array(all_images_as_array_test)
#print(len(x_train))
x_test_test_data = np.reshape(x_test_test_data, (x_test_test_data.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_test_test_data)
x_test_test_data = scaler.transform(x_test_test_data)


# In[50]:


# import y label test
test_dataFrame = pd.read_csv('Datasets\\cartoon_set_test\\labels.csv',header = None, prefix="data")
test_dataFrame['data1']= test_dataFrame['data0'].str.split('\t')
test_df3 = pd.DataFrame(test_dataFrame['data1'].to_list(), columns=['image_number','eye color','face shape','file name'])
test_df3['eye color'] = test_df3['eye color'].replace(['-1'], '0')
test_df3 = test_df3.sort_values(by ='image_number')
test_df3 = test_df3.drop(0)
test_df3['eye color'] = pd.to_numeric(test_df3['eye color'])
#y_test = test_df3['eye color']




# In[51]:


# filter for the same as the non-sunglasses images 
df_y = test_df3[test_df3['file name'].isin(keep_array_test)]

y_test_test_data = df_y['eye color']


# In[52]:


len(y_test_test_data),len(x_test_test_data)


# In[53]:


# Make predictions
predictions = models.predict(x_test_test_data)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test_test_data)
print(accuracy)


# In[54]:


# plot confusion matrix 
plot_confusion_matrix(models, x_test_test_data, y_test_test_data, cmap=plt.cm.Blues)  
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




