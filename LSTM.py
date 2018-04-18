
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = open("ner.txt",'r').readlines()
data = [i.strip().split(' ') for i in data]


# In[2]:


sentences = []
l = []
words = set()
tags = set()
for i in data:
    if i == ['']:
        sentences.append(l)
        l = []
    else:
        tags.add(i[1])
        words.add(i[0])
        l.append(tuple(i))


# In[3]:


tags = list(tags)
words = list(words)
words.append("ENDPAD")
tags


# In[4]:


n_words = len(words); n_words


# In[5]:


n_tags = len(tags); n_tags


# In[6]:


len(sentences)


# In[9]:


max_len = 50
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
tag2idx


# In[10]:


word2idx["assessed"]


# In[11]:


tag2idx["D"]


# In[12]:


from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]


# In[13]:


X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)


# In[14]:


X[1]


# In[15]:


y = [[tag2idx[w[1]] for w in s] for s in sentences]


# In[16]:


y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[17]:


y[-2]


# In[18]:


from keras.utils import to_categorical


# In[19]:


y = [to_categorical(i, num_classes=n_tags) for i in y]
y[-2]


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
len(X_te)


# In[22]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


# In[48]:


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=100, input_length=max_len)(input)
#model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer


# In[49]:


model = Model(input, out)


# In[50]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[51]:


history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)


# In[52]:


hist = pd.DataFrame(history.history)


# In[54]:


plt.figure(figsize=(12,12))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

