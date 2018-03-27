from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

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

model = load_model('best_word_params.hdf5')
# load the mapping
mapping = load(open('mapping_word.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 179 , 'Your', 5))
