
# coding: utf-8

# In[2]:


import nltk
import re
import numpy as np
from nltk.corpus import gutenberg
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from pickle import dump

# In[3]:
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text


# In[4]:


def get_source_text():
    #for key in gutenberg.fileids():
    sentence_vector = gutenberg.sents("carroll-alice.txt")
    #length[key] =  len(sentence_vector)
    length = len(sentence_vector)
    global_text = []
    for sentence in sentence_vector[:int(length)]:
        text = " <s> " + ' '.join(re.compile(r"\w+").findall(' '.join(sentence))).lower() +" </s> "
        global_text.append(text)
    return global_text


# In[5]:


data = get_source_text()
# print(data)
op = ""
for line in data:
    op+=line
print op
data = op


# In[6]:


# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])
dump(tokenizer, open("mapping_word.pkl", 'wb'))

# In[7]:


# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[ ]:


sequences = list()
for line in data.split('</s>  <s>'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)


# In[ ]:


# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]


# In[66]:


# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)


# In[67]:


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(75))
#model.add(LSTM(128, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
filepath = "best_word_params.hdf5"

# In[68]:

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1,save_best_only = True, mode = max)
callbacks_list = [checkpoint]
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y,validation_split = 0.25, epochs=250, verbose=1,batch_size = 2000,callbacks = callbacks_list)

model.save('model_word.h5')
# In[12]:


# evaluate model
#print(generate_seq(model, tokenizer, max_length-1, 'My', 9))

