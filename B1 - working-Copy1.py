#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import os 
import numpy as np
from PIL import Image
from sklearn import preprocessing
from pathlib import Path
from sklearn.model_selection import train_test_split


# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[48]:


#import conusion matrix (plot)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# In[31]:


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


# In[32]:


# import y label train
dataFrame = pd.read_csv('Datasets\\cartoon_set\\labels.csv',header = None, prefix="data")
dataFrame['data1']= dataFrame['data0'].str.split('\t')
df3 = pd.DataFrame(dataFrame['data1'].to_list(), columns=['image_number','eye color','face shape','file name'])
df3['face shape'] = df3['face shape'].replace(['-1'], '0')
  
df3 = df3.sort_values(by ='image_number')
df3 = df3.drop(0)
df3['face shape'] = pd.to_numeric(df3['face shape'])
y_train = df3['face shape']


# In[ ]:





# In[ ]:





# In[33]:


#split data 
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2) 


# In[34]:


print(len(x_train),len(y_train),len(x_test),len(y_test))


# # Cross Validation

# In[35]:


CV_df = pd.DataFrame({"METHOD":[],"MEAN":[],
                         "STD":[]})
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[36]:


# create model
model= LogisticRegression()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_LR = {'Method':'Logistic regression','MEAN':scores.mean(),'STD':scores.std()}


# In[37]:


# support vecotr machine 
model = LinearSVC()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_SV = {'Method':'Support Vector','MEAN':scores.mean(),'STD':scores.std()}


# In[38]:


model = DecisionTreeClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_DT = {'Method':'Decision Tree','MEAN':scores.mean(),'STD':scores.std()}


# In[39]:


model =RandomForestClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_RF = {'Method':'Random Forest','MEAN':scores.mean(),'STD':scores.std()}


# In[40]:


model = KNeighborsClassifier()
# evaluate model
scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
CV_df_KN = {'Method':'K-Nearest Neighbors','MEAN':scores.mean(),'STD':scores.std()}


# In[41]:


list_of_dict = CV_df_LR, CV_df_SV, CV_df_DT,CV_df_RF,CV_df_KN
# Create Dataframe from list of dictionaries and
# pass another list as index
df = pd.DataFrame(list_of_dict)
df


# In[ ]:





# # Hyperparameter 

# In[19]:



# Create the parameter grid based on the results of random search 
param_grid = {'C': [0.1,1, 10, 100]
   }

# Create a based model
sv =LinearSVC()
grid_search = GridSearchCV(estimator = sv, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[20]:


# Fit the grid search to the data
grid_search.fit(x_train, y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_


# In[21]:


best_grid


# In[50]:


# result of CV into tabel 
import pandas as pd

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(
    results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
).rename_axis("kernel")
results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]


# In[51]:


# extract paramets and test scores 
r_df = results_df[['param_C','split0_test_score','split1_test_score' , 'split2_test_score']]


# In[52]:


# plot test scores 
df = r_df.set_index('param_C')
df.T.boxplot()


# In[ ]:





# In[ ]:





# In[43]:


#load classifier 
models =LinearSVC(C=1)
# Fit the classifier
models.fit(x_train, y_train)
    
# Make predictions
predictions = models.predict(x_test)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test)
print(accuracy)


# In[ ]:





# # Testing 

# In[44]:


# import y label test
test_dataFrame = pd.read_csv('Datasets\\cartoon_set_test\\labels.csv',header = None, prefix="data")
test_dataFrame['data1']= test_dataFrame['data0'].str.split('\t')
test_df3 = pd.DataFrame(test_dataFrame['data1'].to_list(), columns=['image_number','eye color','face shape','file name'])
test_df3['face shape'] = test_df3['face shape'].replace(['-1'], '0')
test_df3 = test_df3.sort_values(by ='image_number')
test_df3 = test_df3.drop(0)
test_df3['face shape'] = pd.to_numeric(test_df3['face shape'])
y_test_test_data = test_df3['face shape']
len(y_test_test_data)



# In[45]:


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
x_test_test_data = get_images('Datasets\\cartoon_set_test\\img\\')
x_test_test_data = np.reshape(x_test_test_data, (x_test_test_data.shape[0], -1))
scaler = preprocessing.StandardScaler().fit(x_test_test_data)
x_test_test_data = scaler.transform(x_test_test_data)


# In[46]:


# Make predictions
predictions = models.predict(x_test_test_data)

# Calculate metrics
accuracy= accuracy_score(predictions, y_test_test_data)
print(accuracy)


# # Confusion matrix  

# In[49]:


# plot confusion matrix 
plot_confusion_matrix(models, x_test_test_data, y_test_test_data, cmap=plt.cm.Blues)  
plt.show()


# In[ ]:





# In[ ]:





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




