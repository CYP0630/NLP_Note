import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Activation, Flatten,Dropout,Embedding, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.initializers import Constant

# Hyper-parameter Setup
max_length = 1000
EMBEDDING_DIM = 100
NUM_FILTERS = 12

# Input File
# Some codes in this function refers to this Github Repo:
# Reference 1: https://github.com/Fight-hawk/TextCNN-keras
def load_data_and_labels():
    positive_examples = open("positive.review", encoding="ISO-8859-1")
    positive_examples = [s.strip() for s in positive_examples.readlines()]
    negative_examples = open("negative.review", encoding="ISO-8859-1")
    negative_examples = [s.strip() for s in negative_examples.readlines()]

    # Positive Reviews
    positive_texts = list()
    for i in range(len(positive_examples)):
        if positive_examples[i] == '<review_text>':
            m = 1
            while positive_examples[i + m] != '</review_text>':
                m += 1
            positive_text = ''.join(positive_examples[i + 1:i + m])
            positive_texts.append(positive_text)

    # Negative Reviews
    negative_texts = list()
    for j in range(len(negative_examples)):
        if negative_examples[j] == '<review_text>':
            n = 1
            while negative_examples[j + n] != '</review_text>':
                n += 1
            negative_text = ''.join(negative_examples[j + 1:j + n])
            negative_texts.append(negative_text)

    # Generate labels
    positive_labels = np.ones((len(positive_texts),))
    negative_labels = np.zeros((len(negative_texts),))
    positive_data = pd.DataFrame({'review': positive_texts, 'label': positive_labels})
    negative_data = pd.DataFrame({'review': negative_texts, 'label': negative_labels})
    data = pd.concat([positive_data, negative_data])

    return data

# Tokenize: transfer word by index and count vocabulary size.
def token(train_sentence, test_sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentence)
    x_train = tokenizer.texts_to_sequences(train_sentence)
    x_test = tokenizer.texts_to_sequences(test_sentence)
    word_index = tokenizer.word_index
    vocabulary_size = len(tokenizer.word_index) + 1
    return x_train, x_test, word_index, vocabulary_size


# Build CNN Model
def text_cnn(max_length, num_words, EMBEDDING_DIM, num_filters):
    sequence_input = Input(shape=(max_length,))
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)(sequence_input)

    conv1 = Conv1D(filters=num_filters, kernel_size=3, activation='relu')(embedding_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    flat1 = Flatten()(pool1)

    conv2 = Conv1D(filters=num_filters, kernel_size=4, activation='relu')(embedding_layer)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    flat2 = Flatten()(pool2)

    conv3 = Conv1D(filters=num_filters, kernel_size=5, activation='relu')(embedding_layer)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])
    drop = Dropout(0.8)(merged)
    dense1 = Dense(10, activation='relu')(drop)
    output = Dense(2, activation='softmax')(dense1)
    model = Model(inputs=[sequence_input], outputs=output)

    return model

# Data Pre-processing
data = load_data_and_labels()
review = data['review']
labels = np.array(data['label'])
sentence = review.values

review_train, review_test, y_train, y_test = train_test_split(sentence, labels, test_size=0.1, random_state=3)
x_train, x_test, word_index, vocabulary_size = token(review_train, review_test)
x_train = pad_sequences(x_train, maxlen= max_length, padding='post')
x_test = pad_sequences(x_test, maxlen= max_length, padding='post')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('vocabulary_size:', vocabulary_size)
print('Train samples:', len(x_train))
print('Test samples:', len(x_test))

# GLOVE Pre-Trained Model
# Reference 2: https://keras.io/examples/pretrained_word_embeddings/
print('Loading GLOVE pre-trained model...')
embeddings_index = {}
with open(os.path.join('glove.6B.100d.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

num_words = min(vocabulary_size, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= vocabulary_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Training Part
model = text_cnn(max_length, num_words, EMBEDDING_DIM, NUM_FILTERS)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=16, epochs=100, validation_split=0.1)

# Evaluate Model Performance
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %f' %(accuracy*100))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Val Loss')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.show()
