import pandas as pd
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from nltk.corpus import stopwords
import re
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from keras.utils import np_utils
from time import time

english_stopwords = set(stopwords.words('english'))
categories = ['Democrat', 'Republican']
tweetsdf = pd.read_csv('democratvsrepublicantweets/ExtractedTweets.csv')
handlesdf = pd.read_csv('democratvsrepublicantweets/TwitterHandles.csv')
raw_tweets = tweetsdf['Tweet']
parties = tweetsdf['Party']
y = 1.0 - np.asarray(parties == 'Democrat')

cv = CountVectorizer(stop_words='english', strip_accents='ascii')
bag_of_words = cv.fit_transform([re.sub(r'https?://[^\s]+', '', tweet) for tweet in raw_tweets])

vocab = {}
for feature, freq in zip(cv.get_feature_names(), bag_of_words.sum(axis=0).tolist()[0]):
    if freq > 10:
        vocab[feature] = freq

vocabulary = list(vocab.keys())
vocabulary_size = len(vocabulary) + 1

word_to_int = {word: i + 1 for i, word in enumerate(vocabulary)}
int_to_word = {i + 1: word for i, word in enumerate(vocabulary)}

encoded_tweets =\
    [[word_to_int.get(word, 0) for word in word_tokenize(re.sub(r'https?://[^\s]+', '', tweet).lower())
      if word in word_to_int]
     for tweet in raw_tweets]

max_tweet_len = max([len(encoded_tweet) for encoded_tweet in encoded_tweets])

X = []
for encoded_tweet in encoded_tweets:
    v = np.zeros(max_tweet_len)
    v[0:len(encoded_tweet)] = encoded_tweet
    X.append(v)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def model6(vocabulary_size, max_tweet_len):
    input_layer = Input(shape=(max_tweet_len,))
    x = Embedding(output_dim=100, input_dim=vocabulary_size, input_length=max_tweet_len)(input_layer)
    x = Conv1D(128, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


model = model6(vocabulary_size, max_tweet_len)
model.summary()

filepath = 'word-embedding1.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
callbacks_list = [checkpoint, tensorboard]
model.fit(X_train,
          np_utils.to_categorical(y_train),
          epochs=100,
          batch_size=1000,
          callbacks=callbacks_list)

predictions = np.argmax(model.predict(X_test), axis=1)
performance = [accuracy_score(y_test, predictions),
               precision_score(y_test, predictions),
               recall_score(y_test, predictions),
               f1_score(y_test, predictions)]
cm = confusion_matrix(y_test, predictions)
