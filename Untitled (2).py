
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[5]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
df = pd.read_csv(
    filepath_or_buffer='train.dat',
    header=None, 
    sep='\n')

for i in stop :
    df = df.replace(to_replace=r'\b%s\b'%i, value="",regex=True)
# separate names from classes
print(type(df))
vals = df.loc[:,:].values
names = []
cls =[]

for s in vals:
    temp = s[0].split("\t")
    cls.append(temp[0])
    names.append(temp[1])
    


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
count_vect = CountVectorizer(cls)
X_train_counts = count_vect.fit_transform(names)
X_train_counts.shape


# In[7]:


from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[8]:


from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
clf = MultinomialNB().fit(X_train_tfidf, cls)


# In[9]:


df = pd.read_csv(
    filepath_or_buffer='test.dat', 
    header=None, 
    sep='\n')
for i in stop :
    df = df.replace(to_replace=r'\b%s\b'%i, value="",regex=True)
print(type(df))
# separate names from classes
vals1 = df.loc[:,:].values
names1 = []
cls1 = []
i = 0
for s in vals1:
    names1.append(s[0])


# In[10]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
 ])
text_clf = text_clf.fit(names, cls)
predicted = text_clf.predict(names1)
fp = open("result_1.dat","w+")
fp.write("\n".join(predicted))
#print("Accuracy: {0:.4f}".format(metrics.accuracy_score(cls1, predicted)))


# In[11]:


fp = open("result_1.dat", "w+")
fp.write("\n".join(predicted))


# In[12]:


from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 6))),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, max_iter=5, random_state=42)),
 ])
text_clf_svm.fit(names, cls)
predicted_svm = text_clf_svm.predict(names1)

#print("Accuracy: {0:.4f}".format(metrics.accuracy_score(cls1, predicted_svm)))


# In[13]:


print("\n".join(predicted_svm[:5]))
fp1 = open("result2.dat", "w+")
fp1.write("\n".join(predicted_svm[0:]))

