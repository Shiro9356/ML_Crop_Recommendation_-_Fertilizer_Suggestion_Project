#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing the dataset
data = pd.read_csv('f2.csv')
data.head()


# In[3]:


data.info()


# In[4]:


#changing the column names
data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'},inplace=True)


# In[5]:


#checking unique values
data.nunique()


# In[6]:


#checking for null values
data.isna().sum()


# In[7]:


data['Fertilizer'].unique()


# In[8]:


data['Crop_Type'].unique()


# In[9]:


#statistical parameters
data.describe(include='all')


# In[10]:


plt.figure(figsize=(13, 5))
sns.set(style="whitegrid")
sns.countplot(data=data, x='Crop_Type')
plt.title('Count Plot for Crop_Type')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.show()


# In[11]:


#first 4 crop types
part1_data = data[data['Crop_Type'].isin(data['Crop_Type'].value_counts().index[:4])]

# Create the first countplot
plt.figure(figsize=(10, 4))
sns.set(style="whitegrid")
sns.countplot(data=part1_data, x='Crop_Type', hue='Fertilizer', width=0.8, palette='Set2')
plt.title('First 4 Crop Types')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.legend(title='Fertilizer')
plt.xticks(rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()


# In[12]:


# Split the data into three parts: next 4 crop types
part2_data = data[data['Crop_Type'].isin(data['Crop_Type'].value_counts().index[4:8])]

# Create the second countplot
plt.figure(figsize=(8, 4))
sns.set(style="whitegrid")
sns.countplot(data=part2_data, x='Crop_Type', hue='Fertilizer', width=0.8, palette='Set2')
plt.title('Next 4 Crop Types')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.legend(title='Fertilizer')
plt.xticks(rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()


# In[13]:


# Split the data into three parts: remaining crop types
part3_data = data[data['Crop_Type'].isin(data['Crop_Type'].value_counts().index[8:13])]

# Create the third countplot
plt.figure(figsize=(8, 4))
sns.set(style="whitegrid")
sns.countplot(data=part3_data, x='Crop_Type', hue='Fertilizer', width=0.8, palette='Set2')
plt.title('Remaining Crop Types')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.legend(title='Fertilizer')
plt.xticks(rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()


# In[15]:


# Import necessary libraries
import seaborn as sns
import numpy as np

# Select only numeric columns from the dataframe
numeric_data = data.select_dtypes(include=[np.number])

# Heatmap for Correlation Analysis
sns.heatmap(numeric_data.corr(), annot=True)


# #here is no such correlation between any of variables.. 

# In[16]:


#encoding the labels for categorical variables
from sklearn.preprocessing import LabelEncoder
#it  transforming non-numeric data into a numeric format


# In[17]:


#encoding Soil Type variable
encode_soil = LabelEncoder()

#fitting the label encoder
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

#creating the DataFrame
Soil_Type = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
Soil_Type = Soil_Type.set_index('Original')
Soil_Type


# In[18]:


#encoding Crop Type variable
encode_crop = LabelEncoder()

#fitting the label encoder
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

#creating the DataFrame
Crop_Type = pd.DataFrame(zip(encode_crop.classes_,encode_crop.transform(encode_crop.classes_)),columns=['Original','Encoded'])
Crop_Type = Crop_Type.set_index('Original')
Crop_Type


# In[19]:


#encoding Fertilizer variable
encode_ferti = LabelEncoder()

#fitting the label encoder
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

#creating the DataFrame
Fertilizer = pd.DataFrame(zip(encode_ferti.classes_,encode_ferti.transform(encode_ferti.classes_)),columns=['Original','Encoded'])
Fertilizer = Fertilizer.set_index('Original')
Fertilizer


# In[20]:


#splitting the data into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer',axis=1),data.Fertilizer,test_size=0.2,random_state=1)
print('Shape of Splitting :')
print('x_train = {}, y_train = {}, x_test = {}, y_test = {}'.format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))


# # here 20% of the data should be used for testing (evaluation), and the remaining 80% is used for training
# #x_train and x_test = contain the features (independent variables) used for training and testing the model
# #y_train and y_test = contains the labels(dependent variable) used for training and testing the model.
# 

# In[21]:


x_train.info()


# In[22]:


acc = [] # TEST
model = []
acc1=[] # TRIAN


# Logistic regression model

# In[24]:


# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Fit the Decision Tree model
ds = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
ds.fit(x_train, y_train)

# Predict on the test set
predicted_values = ds.predict(x_test)

# Ensure the length of y_test matches predicted_values
if len(y_test) == len(predicted_values):
    x = metrics.accuracy_score(y_test, predicted_values)
    acc.append(x)
    model.append('Decision Tree')
    
    # Print accuracy on test set
    print("DecisionTree's Accuracy on Test Set is: ", x * 100)
    
    # Classification report for test set
    print(classification_report(y_test, predicted_values))
else:
    print(f"Error: The number of test samples ({len(y_test)}) does not match the number of predicted values ({len(predicted_values)}).")

# Predict on the train set
predicted_values_train = ds.predict(x_train)

# Ensure the length of y_train matches predicted_values_train
if len(y_train) == len(predicted_values_train):
    y = metrics.accuracy_score(y_train, predicted_values_train)
    acc1.append(y)
    
    # Print accuracy on train set
    print("DecisionTree's Accuracy on Train Set is: ", y * 100)
else:
    print(f"Error: The number of train samples ({len(y_train)}) does not match the number of predicted values on train set ({len(predicted_values_train)}).")


# In[26]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Initialize the Naive Bayes model
NaiveBayes = GaussianNB()

# Fit the model on the training data
NaiveBayes.fit(x_train, y_train)

# Predict on the test set
predicted_values = NaiveBayes.predict(x_test)

# Check if the length of y_test matches predicted_values
if len(y_test) == len(predicted_values):
    # Calculate accuracy score on the test set
    x = metrics.accuracy_score(y_test, predicted_values)
    acc.append(x)
    print("Naive Bayes's Accuracy on Test Set is: ", x * 100)
    
    # Print classification report for the test set
    print(classification_report(y_test, predicted_values))
else:
    print(f"Error: The number of test samples ({len(y_test)}) does not match the number of predicted values ({len(predicted_values)}).")

# Predict on the training set
predicted_values_train = NaiveBayes.predict(x_train)

# Check if the length of y_train matches predicted_values_train
if len(y_train) == len(predicted_values_train):
    # Calculate accuracy score on the training set
    y = metrics.accuracy_score(y_train, predicted_values_train)
    acc1.append(y)
    
    # Print accuracy on train set
    print("Naive Bayes's Accuracy on Train Set is: ", y * 100)
else:
    print(f"Error: The number of train samples ({len(y_train)}) does not match the number of predicted values on train set ({len(predicted_values_train)}).")


# In[27]:


from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Normalize the data using MinMaxScaler
norm = MinMaxScaler().fit(x_train)
X_train_norm = norm.transform(x_train)
X_test_norm = norm.transform(x_test)

# Initialize the SVM model with polynomial kernel
SVM = SVC(kernel='poly', degree=3, C=1)

# Train the model on the normalized training data
SVM.fit(X_train_norm, y_train)

# Predict on the test set
predicted_values = SVM.predict(X_test_norm)

# Check if the lengths of y_test and predicted_values match
if len(y_test) == len(predicted_values):
    # Calculate the accuracy score on the test set
    x = metrics.accuracy_score(y_test, predicted_values)
    acc.append(x)
    print("SVM's Accuracy on Test Set is: ", x * 100)

    # Print classification report for the test set
    print(classification_report(y_test, predicted_values))
else:
    print(f"Error: The number of test samples ({len(y_test)}) does not match the number of predicted values ({len(predicted_values)}).")

# Predict on the training set
predicted_values_train = SVM.predict(X_train_norm)

# Check if the lengths of y_train and predicted_values_train match
if len(y_train) == len(predicted_values_train):
    # Calculate accuracy on the training set
    y = metrics.accuracy_score(y_train, predicted_values_train)
    acc1.append(y)
    print("SVM's Accuracy on Train Set is: ", y * 100)
else:
    print(f"Error: The number of train samples ({len(y_train)}) does not match the number of predicted values on train set ({len(predicted_values_train)}).")


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Initialize the Logistic Regression model
LogReg = LogisticRegression(random_state=2)

# Train the model
LogReg.fit(x_train, y_train)

# Predict on the test set
predicted_values = LogReg.predict(x_test)

# Check if the lengths of y_test and predicted_values match
if len(y_test) == len(predicted_values):
    # Calculate the accuracy score
    x = metrics.accuracy_score(y_test, predicted_values)
    acc.append(x)

    # Print classification report
    print("Logistic Regression's Accuracy on Test Set is: ", x * 100)
    print(classification_report(y_test, predicted_values))
else:
    print(f"Error: The number of test samples ({len(y_test)}) does not match the number of predicted values ({len(predicted_values)}).")

# Predict on the training set (for comparison)
predicted_values_train = LogReg.predict(x_train)

# Check if the lengths of y_train and predicted_values_train match
if len(y_train) == len(predicted_values_train):
    # Calculate accuracy on the training set
    y = metrics.accuracy_score(y_train, predicted_values_train)
    acc1.append(y)
    print("Logistic Regression's Accuracy on Train Set is: ", y * 100)
else:
    print(f"Error: The number of train samples ({len(y_train)}) does not match the number of predicted values on train set ({len(predicted_values_train)}).")


# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Initialize the RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)

# Print sizes of train and test sets to check for consistency
print(f"Training data size: {len(x_train)}, Test data size: {len(x_test)}")
print(f"Training labels size: {len(y_train)}, Test labels size: {len(y_test)}")

# Train the model
RF.fit(x_train, y_train)

# Predict on the test set
predicted_values = RF.predict(x_test)

# Check if the lengths match between y_test and predicted_values
print(f"Length of y_test: {len(y_test)}")
print(f"Length of predicted_values: {len(predicted_values)}")

# Calculate accuracy for the test set
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)

# Predict on the training set (for comparison)
predicted_values_train = RF.predict(x_train)

# Calculate accuracy for the train set
y = metrics.accuracy_score(y_train, predicted_values_train)
acc1.append(y)

# Append model to list and print accuracies
model.append('RF')
print("Random Forest's Accuracy on Test Set is: ", x)
print("Random Forest's Accuracy on Train Set is: ", y)

# Print classification report for the test set
print(classification_report(y_test, predicted_values))


# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Initialize the RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)

# Print sizes of train and test sets to check for consistency
print(f"Training data size: {len(x_train)}, Test data size: {len(x_test)}")
print(f"Training labels size: {len(y_train)}, Test labels size: {len(y_test)}")

# Train the model
RF.fit(x_train, y_train)

# Predict on the test set
predicted_values = RF.predict(x_test)

# Check if the lengths match between y_test and predicted_values
print(f"Length of y_test: {len(y_test)}")
print(f"Length of predicted_values: {len(predicted_values)}")

# Calculate accuracy for the test set
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)

# Predict on the training set (for comparison)
predicted_values_train = RF.predict(x_train)

# Calculate accuracy for the train set
y = metrics.accuracy_score(y_train, predicted_values_train)
acc1.append(y)

# Append model to list and print accuracies
model.append('RF')
print("Random Forest's Accuracy on Test Set is: ", x)
print("Random Forest's Accuracy on Train Set is: ", y)

# Print classification report for the test set
print(classification_report(y_test, predicted_values))


# In[31]:


from sklearn.model_selection import cross_val_score

score = cross_val_score(RF,data,data.Fertilizer,cv=5)
print("Cross-validation score of RF is:",score)
score = cross_val_score(LogReg,data,data.Fertilizer,cv=5)
print("Cross-validation score of LogReg is:",score)
score = cross_val_score(SVM,data,data.Fertilizer,cv=5)
print("Cross-validation score of SVM is:",score)
score = cross_val_score(NaiveBayes,data,data.Fertilizer,cv=5)
print("Cross-validation score of NaiveBayes is:",score)
score = cross_val_score(ds, data, data.Fertilizer,cv=5)
print("Cross-validation score of ds is:",score)


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize lists for storing accuracy values and model names
acc = []
model = []

# Logistic Regression
LogReg = LogisticRegression(random_state=2)
LogReg.fit(x_train, y_train)
predicted_values = LogReg.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
predicted_values = LogReg.predict(x_train)
y = metrics.accuracy_score(y_train, predicted_values)
acc.append(y)
model.append('Logistic Regression')
model.append('Logistic Regression (Train)')

# Random Forest
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train, y_train)
predicted_values = RF.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
predicted_values = RF.predict(x_train)
y = metrics.accuracy_score(y_train, predicted_values)
acc.append(y)
model.append('Random Forest')
model.append('Random Forest (Train)')

# Support Vector Classifier (SVC)
svc = SVC(random_state=0)
svc.fit(x_train, y_train)
predicted_values = svc.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
predicted_values = svc.predict(x_train)
y = metrics.accuracy_score(y_train, predicted_values)
acc.append(y)
model.append('SVC')
model.append('SVC (Train)')

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predicted_values = knn.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
predicted_values = knn.predict(x_train)
y = metrics.accuracy_score(y_train, predicted_values)
acc.append(y)
model.append('KNN')
model.append('KNN (Train)')

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
predicted_values = dt.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
predicted_values = dt.predict(x_train)
y = metrics.accuracy_score(y_train, predicted_values)
acc.append(y)
model.append('Decision Tree')
model.append('Decision Tree (Train)')

# Plotting the accuracy comparison
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')

# Ensure that the lengths of acc and model match before plotting
sns.barplot(x=acc, y=model, palette='dark')
plt.show()


# In[39]:


#pickling the file
import pickle
pickle_out = open('classifier.pkl','wb')
pickle.dump(RF,pickle_out)
pickle_out.close()


# In[40]:


model = pickle.load(open('classifier.pkl','rb'))
model.predict([[34,67,62,0,1,7,0,30]])


# In[41]:


model = pickle.load(open('classifier.pkl','rb'))
model.predict([[25,78,43,4,1,22,26,38]])


# In[42]:


#pickling the file
import pickle
pickle_out = open('fertilizer.pkl','wb')
pickle.dump(encode_ferti,pickle_out)
pickle_out.close()


# In[43]:


ferti = pickle.load(open('fertilizer.pkl','rb'))
ferti.classes_[1]


# In[ ]:




