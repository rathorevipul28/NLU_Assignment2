##D1-TRAIN + D2-TRAIN
###D2-TRAIN
from nltk.corpus import gutenberg
import re
import nltk
import sys
import os
import re
import nltk
import math
from random import randint
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
lem = WordNetLemmatizer()
f=open("/home/sclab/Desktop/assign1/gutenberg/cats1.txt","r")
lem = WordNetLemmatizer()
new_token2=[]  # check the open source for writing an array that can store n values i.e. list
main_cat1=[]
main1=[]
cat_dict1= {}
for i in f.readlines():
    
    i = word_tokenize(i)
    #print(i[0])
    main1.append(i)
#print(main[1])
#print(len(main))
#print(main)
for k in range(0,len(main1),1):
    main_cat1.append(main1[k][1])
    if main1[k][1] in cat_dict1:
        cat_dict1[main1[k][1]] += 1
    else:
        cat_dict1[main1[k][1]] = 1

        
#L='news'
#print(cat_dict[L])
print(len(cat_dict1))
print(cat_dict1['gutenberg'])
print(cat_dict1)

length=0
lent=0
val1=0
train_sample1=[]
test_sample1=[]
dev_sample1=[]
tok1 = []
pos1 = []
tok_dict1 = {}
train_data = list()
test_data = list()
data = list()
all_words = set()

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
size1=[]   ## that keeps the track of length of each line by line readings used later
#for val1 in range(0,len(main1),val1):

R=main1[lent][1]
    #print(R)
length1=cat_dict1[R]
    #print(length)

p=1
val=0
for val in range(val,length1,1):
    emma = open("/home/sclab/Desktop/assign1/gutenberg/"+main1[val][0]+".txt")
    #emma = gutenberg.sents()
    sentence = []
    for line in emma:
        #print(sent)
        words = line.split(' ')
        for word in words:
            if word != '\n' and word != '':
                if word[-1] == '.' and word not in ['Mr.','Mrs.'] :
                    sentence.append(re.sub(r'[^\w\s]','',word.strip().lower()))
                    data.append(' '.join(sentence))
                    #print(sentence)
                    sentence = []
                else:
                    sentence.append(re.sub(r'[^\w\s]','',word.strip().lower()))
            #print(data)
            #print(p)
            #p=p+1
    break
print(len(data))
length_1=int(len(data)*0.4)
for i in range(0,length_1):
    if i < 0.7 * length_1:
        train_data.append(data[i])
    else:
        test_data.append(data[i])
        

print(len(train_data))
print(len(test_data))
#train_data = [i for i in train_data if len(i) > 6]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)
encoded = [tokenizer.texts_to_sequences([line])[0] for line in train_data]				
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create line-based sequences
sequences = list()
for line in encoded:	
	for i in range(1, len(line)):
		sequence = line[:i+1]
		sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
max_length = max([len(seq) for seq in encoded])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = np.array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
# define model
with tf.device('/gpu:0'):
    model = Sequential()
    model.add(Embedding(vocab_size, 20, input_length=max_length-1))
    model.add(LSTM(20))	
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import numpy as np
mask = np.random.choice(len(test_data),2,replace=False)
mask=list(mask)
print(mask)
t=(test_data[i] for i in mask)
dataset=list(t)
x=dataset[0]
p=dataset[1]
# fit network
model.fit(X, y, epochs=20, verbose=2,batch_size=1000)
print(generate_seq(model, tokenizer, max_length-1, 'besides', 10))
print(generate_seq(model, tokenizer, max_length-1, 'opinion', 10))
print(dataset[0])
print(dataset[1])
