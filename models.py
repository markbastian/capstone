from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Model


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
    """A convolutional network with no embeddings."""
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
    """A convolutional network with no embeddings."""
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
