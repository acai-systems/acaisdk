import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)


# Removing the stopwords from text
def remove_stopwords(text, stop):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


# Removing the noisy text
def denoise_text(text):
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text, stop)
    return text

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


input_dir = sys.argv[1]
output_dir = sys.argv[2]
embedding_file = sys.argv[3]  # 'glove.twitter.27B.100d.txt'

true = pd.read_csv(input_dir + "True.csv")
false = pd.read_csv(input_dir + "Fake.csv")
true['category'] = 1
false['category'] = 0

df = pd.concat([true, false])
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

df['text'] = df['text'].apply(denoise_text)

# Splitting & Training
x_train, x_test, y_train, y_test = train_test_split(df.text, df.category, random_state=0)
max_features = 10000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)
tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = keras.utils.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(x_test)
X_test = keras.utils.pad_sequences(tokenized_test, maxlen=maxlen)
EMBEDDING_FILE = input_dir + embedding_file

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
# change below line if computing normal stats is too slow
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

batch_size = 256
epochs = 10
embed_size = 100

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Defining Neural Network
model = Sequential()
# Non-trainable embeddidng layer
model.add(
    Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))

# LSTM
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
model.add(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=epochs,
                    callbacks=[learning_rate_reduction])

with open(output_dir + '/accuracy.txt', 'w') as f:
    f.write("Accuracy of the model on Training Data is - " + str(model.evaluate(x_train, y_train)[1] * 100) + "%\n")
    f.write("Accuracy of the model on Testing Data is - " + str(model.evaluate(X_test, y_test)[1] * 100) + "%\n")
