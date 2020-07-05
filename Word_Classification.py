#!/usr/bin/env python
# coding: utf-8

# ( 1.1 ) Word Classification with Machine Learing
# 
# In this case I classified given Words in different Languages as English and Turkish.
# The main aim is to predict the Language of a given Word with a Machine Learning algorithm where I used Support Vector Machines,  Naive Bayes and Logistic regression to compare each Algorithm and find the best solution.
# 

# In[1]:


# Importing libraries

# I used the Pandas library to read the csv files, to creade the dataframes
import pandas as pd
# I used the Numpy library to create Numpy arrays for the labels, append the labels to the datas and concatenate two dataset
import numpy as np


# In[2]:


# Reading csv files
data_eng = pd.read_csv(r"C:\Users\user\Desktop\Case\English.csv")
data_tr = pd.read_csv(r"C:\Users\user\Desktop\Case\Turkish.csv")


# In[3]:


# Getting the length of each dataset
len_en = len(data_eng)
len_tr = len(data_tr)

# Creating arrays for labels to each category ( English = 0 , Turkish = 1)
class_en = np.zeros((len_en,1), dtype=np.int64)
class_tr = np.ones((len_tr,1), dtype=np.int64)

# Adding the labels as 2. Column to each dataset
data_eng = np.append(data_eng, class_en, axis=1)
data_tr = np.append(data_tr, class_tr, axis=1)


# In[4]:


# Concatinating the datasets
data = np.concatenate((data_eng, data_tr), axis=0)
data = pd.DataFrame(data)

# Fixing the column names of the dataframe
data = data.rename({0: "word", 1: "language"}, axis='columns')

#Cheking if there is any NaN value
data.isna().sum() #there is only 1 NaN value

# Dropping NaN values
data = data.dropna()


# In[5]:


# Dataset after the data Preparation ( We have both English and Turkish word in one Column and labeled them as 0 and 1)
data


# In[6]:


# Splitting Data and Labels
X = data['word']
y = data['language']
#convering type of y to integer
y=y.astype('int')


# In[7]:


# Splitting the data randomly as train and test
# I used the Sklearn library to use the train test split function, to transform the words to vectors,
#    Train the data with the naive bayes algorithm and to get the accuracy score
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size = 0.20, random_state = 42) 


# In[8]:


# Vectorizing words for the algorithm!!
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)


# In[9]:


# Training model ( Support Vector Machines (Polynomial) )
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
SVM = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto')
SVM.fit(train_vectors,y_train)

# Calculating Accuracy
predictions_SVM = SVM.predict(test_vectors)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test))


# In[10]:


# Training model ( Support Vector Machines (Linear) )
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_vectors,y_train)

# Calculating Accuracy
predictions_SVM = SVM.predict(test_vectors)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test))


# In[11]:


# Training model ( Multinomial Naive Bayes )
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_vectors, y_train)

# Calculating Accuracy
from  sklearn.metrics  import accuracy_score
predicted = clf.predict(test_vectors)
print("Multinomial Naive Bayes Accuracy Score -> ",accuracy_score(y_test,predicted))


# In[12]:


# Training model ( Bernoulli Naive Bayes )
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(train_vectors, y_train)

# Calculating Accuracy
from  sklearn.metrics  import accuracy_score
predicted = clf.predict(test_vectors)
print("Bernoulli Naive Bayes Accuracy Score -> ",accuracy_score(y_test,predicted))


# In[13]:


# Training model ( Logistic Regression )
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(train_vectors, y_train)

# Calculating Accuracy
clf.predict(test_vectors)
print("Logistic Regression Accuracy Score -> ",clf.score(train_vectors,y_train))


# The accuracy of predicting the language of a given word is close to 90% for each algorithm. In this case I could say that,with only one feature the accuracy of a word classification is almost satisfying.
# The weak part of the algorithm is that we just used randomly selected words instead of Texts.
# If we are using texts instead of words which will give use opportunity to use more Natural Language Process techniques we will have a higher accuracy.
