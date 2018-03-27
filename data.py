
# coding: utf-8

# In[1]:


import nltk
import re
import numpy as np
from nltk.corpus import gutenberg
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
import h5py
from keras.models import load_model


# In[2]:


def get_source_text():                                                                                                                                             
    #for key in gutenberg.fileids():                                                                                                                                 
    sentence_vector = gutenberg.sents("carroll-alice.txt")                                                                                                           
    #length[key] =  len(sentence_vector)                                                                                                                             
    length = len(sentence_vector)                                                                                                                                    
    global_text = []                                                                                                                                                 
    for sentence in sentence_vector[:int(length)]:                                                                                                            
        text = "" + ' '.join(re.compile(r"\w+").findall(' '.join(sentence))).lower() +"\n"                                                                                                                                                     
        global_text.append(text)                                                                                                                                     
    return global_text 


# In[3]:


data = get_source_text()
op = ""                                                                                                                                                              
for line in data:                                                                                                                                                    
    op+=line                                                                                                                                                         
print op                                                                                                                                                             
data = op
f = open("data.txt",'w')
f.write(data)
f.close()



