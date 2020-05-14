import re
import numpy as np
import gensim
import keras
import keras.layers as L
from collections import Counter
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt

# Input File
File1 = '28054-0.txt'
File2 = 'pg1661.txt'
File3 = 'pg31100.txt'

# Hyper-parameter Setup
USE_PRE_TRAIN_EMBEDDING = True
EMBEDDING_DIM = 300
DEV_SAMPLE_PERCENTAGE = 0.1
NUM_CLASSES = 3
NUM_FILTERS = 16
FILTER_SIZES = (3, 4, 5)

# Define a function to cleaning text data.
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Loading original file and construct dataset.
def load_data_and_labels(file1, file2, file3):
    # Load data from files
    data1 = list(open(file1, "r", encoding='utf-8').readlines())
    data1 = [s.strip() for s in data1]
    data2 = list(open(file2, "r", encoding='utf-8').readlines())
    data2 = [s.strip() for s in data2]
    data3 = list(open(file3, "r", encoding='utf-8').readlines())
    data3 = [s.strip() for s in data3]

    # Construct datset and Cleaning dataset
    x_text = data1 + data2 + data3
    x_text = [clean_str(sent) for sent in x_text]

    # One-hot embedding for label
    label1 = [[0, 0, 1] for _ in data1]
    label2 = [[0, 1, 0] for _ in data2]
    label3 = [[1, 0, 0] for _ in data3]

    y = np.concatenate([label1, label2, label3], 0)
    return [x_text, y]

# Prepare for 2D conv layer
def as_matrix(sequences, max_len, index2word):
    matrix = np.full((len(sequences), max_len), 0)
    for i, seq in enumerate(sequences):
        row_ix = [index2word.index(w) for w in seq.split(' ')]
        matrix[i, :len(row_ix)] = row_ix
    return matrix

# Using Pre-trained model to embedding words.
def get_pre_train_word2vec(model, index2word, vocab_size):
    embedding_size = model.vector_size
    pre_train_word2vec = dict(zip(model.vocab.keys(), model.vectors))
    word_embedding_2dlist = [[]] * vocab_size    # [vocab_size, embedding_size]
    word_embedding_2dlist[0] = np.zeros(embedding_size)    # assign empty for first word:'PAD'
    pre_count = 0    # vocabulary in pre-train word2vec
    # loop for all vocabulary, note that the first is 'PDA'
    for i in range(1, vocab_size):
        if index2word[i] in pre_train_word2vec:
            word_embedding_2dlist[i] = pre_train_word2vec[index2word[i]]
            pre_count += 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(-0.1, 0.1, embedding_size)
    return np.array(word_embedding_2dlist), pre_count

# Calculates the precision
def precision(model, x_test ,y_true):
    y_true = np.argmax(y_true, axis=1)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis=1)
    true_count = sum(y_true == y_predict)
    return true_count / len(y_true)

# Calculates the recall
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Define Model
def text_cnn(sequence_length, num_classes, vocab_size, embedding_size,
             filter_sizes, num_filters, embedding_matrix, drop_out=0.5):

    # Embedding Layer.
    input_x = L.Input(shape=(sequence_length,), name='input_x')
    #if embedding_matrix is None:
    embedding = L.Embedding(vocab_size, embedding_size, name='embedding')(input_x)
    #else:
    #   embedding = L.Embedding(vocab_size, embedding_size, weights=[embedding_matrix], name='embedding')(input_x)
    expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
    embedding_chars = L.Reshape(expend_shape)(embedding)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv = L.Conv2D(filters=num_filters,
                        kernel_size=[filter_size, embedding_size],
                        strides=1, padding='valid',
                        activation='relu',
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                        bias_initializer=keras.initializers.constant(value=0.1),
                        name=('conv_%d' % filter_size))(embedding_chars)
        max_pool = L.MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
                               strides=(1, 1),
                               padding='valid',
                               name=('max_pool_%d' % filter_size))(conv)
        pooled_outputs.append(max_pool)

    # Combine the pooled features as a dense layer
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = L.Concatenate(axis=3)(pooled_outputs)
    h_pool_flat = L.Reshape([num_filters_total])(h_pool)
    # add dropout
    dropout = L.Dropout(drop_out)(h_pool_flat)

    # output layer
    output = L.Dense(num_classes,
                     kernel_initializer='glorot_normal',
                     bias_initializer=keras.initializers.constant(0.1),
                     activation='softmax',
                     name='output')(dropout)

    model = keras.models.Model(inputs=input_x, outputs=output)

    return model


x_text, y = load_data_and_labels(File1, File2, File3)
print('Total records of the dataset: ', len(x_text))
max_doc_length = max([len(x.split(' ')) for x in x_text])
print("Max document length: ", max_doc_length)


tokens = [t for doc in x_text for t in doc.split(' ')]
print("Total tokens in dataset: ", len(tokens))
counter = Counter(tokens)
index2word = list(counter.keys())
index2word.insert(0, 'PAD')
print("Vocabulary size in dataset: ", len(index2word))

x_matrix = as_matrix(x_text, max_doc_length, index2word)
x_train, x_test, y_train, y_test = train_test_split(x_matrix, y, test_size=DEV_SAMPLE_PERCENTAGE)
print('Train part: ', len(x_train))
print('Test part:', len(x_test))

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
word_embedding, pre_count = get_pre_train_word2vec(word2vec_model, index2word, len(index2word))

cnn = text_cnn(x_train.shape[1], NUM_CLASSES, len(index2word), EMBEDDING_DIM, FILTER_SIZES, NUM_FILTERS, None)
cnn.compile(loss='categorical_crossentropy',
                 optimizer=keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False),
                 metrics=['accuracy', recall_m])
cnn_history = cnn.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
precession_score = precision(cnn, x_test, y_test)
print('Precession Score is:', precession_score)
recall_score = cnn.evaluate(x_test, y_test, verbose=0)
print('Recall Score is:', recall_score)

plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('Training Loss vs Val Loss')
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.show()
