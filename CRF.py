
# coding: utf-8

# In[1]:


import numpy
import nltk
from nltk.tokenize import word_tokenize


# In[2]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[3]:


from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# In[4]:


data = open("ner.txt",'r').readlines()
data = [i.strip().split(' ') for i in data]
len(data)


# In[78]:


get_ipython().run_cell_magic('time', '', "sentences = []\nl = []\nwords = set()\ntags = set()\nfor i in data:\n    if i == ['']:\n        sentences.append(l)\n        l = []\n    else:\n        tags.add(i[1])\n        words.add(i[0])\n        text = word_tokenize(i[0])\n        k = [i[0],nltk.pos_tag(text)[0][1],i[1]]\n        l.append(tuple(k))")


# In[79]:


train_sents = sentences[0:int(0.8*len(sentences))]
test_sents = sentences[int(0.8*len(sentences)):]


# In[80]:


train_sents[0]


# In[141]:


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length()' : len(word),
        'postag': postag,
        #'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.length()' : len(word1),
            '-1:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.length()' : len(word1),
            '+1:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
        
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][1]
        features.update({
            '-2:word.lower()': word1.lower(),
            '-2:word.istitle()': word1.istitle(),
            '-2:word.isupper()': word1.isupper(),
            '-2:word.length()' : len(word1),
            '-2:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS1'] = True
        
    if i < len(sent)-2:
        word1 = sent[i+2][0]
        postag1 = sent[i+2][1]
        features.update({
            '+2:word.lower()': word1.lower(),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.isupper()': word1.isupper(),
            '+2:word.length()' : len(word1),
            '+2:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS1'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,p, label in sent]

def sent2tokens(sent):
    return [token for token,p, label in sent]


# In[142]:


sent2features(train_sents[0])


# Extract features from the data:

# In[143]:


get_ipython().run_cell_magic('time', '', 'X_train = [sent2features(s) for s in train_sents]\ny_train = [sent2labels(s) for s in train_sents]\n\nX_test = [sent2features(s) for s in test_sents]\ny_test = [sent2labels(s) for s in test_sents]')


# In[144]:


get_ipython().run_cell_magic('time', '', "crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs', \n    c1=0.1, \n    c2=0.1, \n    max_iterations=100, \n    all_possible_transitions=True\n)\ncrf.fit(X_train, y_train)")


# In[145]:


labels = list(crf.classes_)
labels


# In[146]:


y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=labels)

