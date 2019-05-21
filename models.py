from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from time import time
import seaborn as sn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plot_confusion_matrix

def plot_results(y_test, predictions):
    objects = ('Accuracy', 'Precision', 'Recall', 'F1')
    y_pos = np.arange(len(objects))
    performance = [accuracy_score(y_test, predictions),
                   precision_score(y_test, predictions),
                   recall_score(y_test, predictions),
                   f1_score(y_test, predictions)]
    cm = confusion_matrix(y_test, predictions)

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Score')
    plt.title('Baseline Performance Metrics')
    plt.show()

    plt.rcParams["figure.figsize"] = (7,7)
    classes=np.array(['D (0)', 'R (1)'])
    plot_confusion_matrix.plot_confusion_matrix(y_test.astype(int), predictions.astype(int), 
                                                classes=classes,
                                                title='Confusion matrix')
    plt.show()
    plot_confusion_matrix.plot_confusion_matrix(y_test.astype(int), predictions.astype(int), 
                                                classes=classes, 
                                                normalize=True,
                                                title='Normalized confusion matrix')
    plt.show()

    return performance, cm


def train(X_train, y_train, model, filepath, num_epochs=100, batch_size=1000):
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    callbacks_list = [checkpoint, tensorboard]
    model.fit(X_train,
              np_utils.to_categorical(y_train),
              epochs=num_epochs,
              batch_size=batch_size,
              callbacks=callbacks_list)


def model1(input_length):
    """A convolutional network with no embeddings."""
    input_layer = Input(shape=(input_length,1))
    x = Conv1D(256, kernel_size=4, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=4, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=4, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'cnn-weights-0.0337.hdf5', model


def model2(input_length):
    """A convolutional network with no embeddings, but adds dropouts."""
    input_layer = Input(shape=(input_length, 1))
    x = Conv1D(256, kernel_size=4, activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'model2-weights.hdf5', model


# Round 3 - Use character embedding
# https://keras.io/getting-started/functional-api-guide/
def model3(vocabulary_size, input_length):
    """A convolutional network with character embeddings."""
    input_layer = Input(shape=(input_length,))
    x = Embedding(output_dim=256, input_dim=vocabulary_size, input_length=input_length)(input_layer)
    x = Conv1D(256, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'model3-weights.hdf5', model


def model4(vocabulary_size, input_length):
    """A convolutional network with character embeddings."""
    input_layer = Input(shape=(input_length,))
    x = Embedding(output_dim=256, input_dim=vocabulary_size, input_length=input_length)(input_layer)
    x = Conv1D(256, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=4, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'model4-weights.hdf5', model


def model5(vocabulary_size, input_length):
    """Embeddings followed by 3 Conv/Max Pooling layers"""
    input_layer = Input(shape=(input_length,))
    x = Embedding(output_dim=256, input_dim=vocabulary_size, input_length=input_length)(input_layer)
    x = Conv1D(64, kernel_size=4, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=4, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=4, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'model5-weights.hdf5', model


# https://medium.com/datadriveninvestor/sentiment-analysis-using-embeddings-f3dd99aeaade
def model6(vocabulary_size, input_length):
    input_layer = Input(shape=(input_length,))
    x = Embedding(output_dim=32, input_dim=vocabulary_size, input_length=input_length)(input_layer)
    x = LSTM(2)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'model6-weights.hdf5', model


def model7(vocabulary_size, max_tweet_len):
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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return 'word-embedding1.hdf5', model


def model8(vocabulary_size, max_tweet_len):
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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return 'word-embedding2.hdf5', model


def model9(vocabulary_size, input_length):
    input_layer = Input(shape=(input_length,))
    x = Embedding(output_dim=32, input_dim=vocabulary_size, input_length=input_length)(input_layer)
    x = LSTM(2)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input_layer, x)
    optimizer = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return 'model6-weights.hdf5', model