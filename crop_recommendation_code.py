#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Importing libraries

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[19]:


PATH = 'Crop_recommendation.csv'
df = pd.read_csv(PATH)


# In[20]:


df.head()


# In[21]:


df.tail()


# In[22]:


df.size


# In[23]:


df.shape


# In[24]:


df.columns


# In[25]:


df['label'].unique()


# In[26]:


df.dtypes


# In[27]:


df['label'].value_counts()


# In[28]:


df1 = df.drop(columns = ['label'])
df1.head()


# In[29]:


sns.heatmap(df1.corr(),annot=True)


# In[30]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']


# In[31]:


# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []


# In[32]:


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[33]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))


# In[34]:


from sklearn.model_selection import cross_val_score


# In[35]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)


# In[36]:


score


# In[37]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
DT_pkl_filename = 'DecisionTree.pkl'
# Open the file to save as pkl file
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
# Close the pickle instances
DT_Model_pkl.close()


# In[40]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[41]:


# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
score


# In[42]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
NB_pkl_filename = 'NBClassifier.pkl'
# Open the file to save as pkl file
NB_Model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
# Close the pickle instances
NB_Model_pkl.close()


# In[43]:


from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(Xtrain,Ytrain)

predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[44]:


# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
score


# Logistic Regression
# 

# In[45]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[46]:


# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
score


# Saving trained Logistic Regression model

# In[47]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
LR_pkl_filename = 'LogisticRegression.pkl'
# Open the file to save as pkl file
LR_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(LogReg, LR_Model_pkl)
# Close the pickle instances
LR_Model_pkl.close()


# Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[49]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score


# Saving trained Random Forest model

# In[50]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


# XGBoost

#  Convert Target Labels to Numeric

# In[60]:


# Import necessary libraries
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report

# Encode the target labels (Ytrain and Ytest) since they contain string labels
label_encoder = LabelEncoder()

# Fit and transform on the training data, and transform on the test data
Ytrain_encoded = label_encoder.fit_transform(Ytrain)
Ytest_encoded = label_encoder.transform(Ytest)

# Initialize the XGBoost classifier
XB = xgb.XGBClassifier()

# Fit the model with the encoded labels
XB.fit(Xtrain, Ytrain_encoded)

# Predict the values
predicted_values = XB.predict(Xtest)

# Calculate accuracy score
x = metrics.accuracy_score(Ytest_encoded, predicted_values)

# Store the accuracy and model name for comparison (if needed later)
acc.append(x)
model.append('XGBoost')

# Output the accuracy
print("XGBoost's Accuracy is: ", x)

# Print classification report using encoded labels
print(classification_report(Ytest_encoded, predicted_values))


# In[62]:


from sklearn.model_selection import cross_val_score
# Encode the target labels (since they are strings)
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Initialize the XGBoost classifier
XB = xgb.XGBClassifier()

# Cross-validation score (XGBoost)
score = cross_val_score(XB, features, target_encoded, cv=5)

# Output the cross-validation scores
score


# In[63]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
XB_pkl_filename = 'XGBoost.pkl'
# Open the file to save as pkl file
XB_Model_pkl = open(XB_pkl_filename, 'wb')
pickle.dump(XB, XB_Model_pkl)
# Close the pickle instances
XB_Model_pkl.close()


# Accuracy Comparison

# In[64]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[65]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# Making a Prediction

# In[66]:


data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)


# In[67]:


data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = RF.predict(data)
print(prediction)


# In[ ]:




