#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np


# In[41]:


data_eng = pd.read_csv(r"C:\Users\user\Desktop\Case\English.csv")
data_tr = pd.read_csv(r"C:\Users\user\Desktop\Case\Turkish.csv")


# In[42]:


#label eklmeke için verisetinin uzunluğunu alıyorum
len_en = len(data_eng)
len_tr = len(data_tr)

#etiketleri oluşturuyorum
class_en = np.zeros((len_en,1), dtype=np.int64)
class_tr = np.ones((len_tr,1), dtype=np.int64)

#etiketleri kelimelerin yanına ekliyorum
data_eng = np.append(data_eng, class_en, axis=1)
data_tr = np.append(data_tr, class_tr, axis=1)


# In[43]:


#iki ayrı verisetini birleştirme
data = np.concatenate((data_eng, data_tr), axis=0)
data = pd.DataFrame(data)

#sütünların isimlerini düzeltme
data = data.rename({0: "word", 1: "language"}, axis='columns')

#NAN değerleri drop ettim
data = data.dropna()


# In[44]:


data


# In[45]:


#verileri ayrıştırma
X = data['word']
y = data['language']
y=y.astype('int')


# In[46]:


#train test split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size = 0.33, random_state = 42) 


# In[47]:


#kelimelerin vectorize edilmesi
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)


# In[48]:


#naive bayes uygulanması
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_vectors, y_train)


# In[49]:


#accuracy hesaplama
from  sklearn.metrics  import accuracy_score
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))


# In[ ]:





# In[ ]:




