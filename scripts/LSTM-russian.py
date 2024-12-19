import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Concatenate
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter

import os
os.environ['TF_TRT_ALLOW_BUILD_AT_RUNTIME'] = '0'
# Parameters
MAX_VOCAB_SIZE = 4096
MAX_SEQ_LEN = 25
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
AUTHOR_EMBEDDING_DIM = 2
BATCH_SIZE = 16
EPOCHS = 5

# 1. Tokenizer
class Tokenizer:
    def __init__(self, max_vocab_size=MAX_VOCAB_SIZE):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<OOV>": 1}
        self.idx2word = {0: "<PAD>", 1: "<OOV>"}

    def fit_on_texts(self, texts):
        word_counts = Counter(word for text in texts for word in text.split())
        most_common = word_counts.most_common(self.max_vocab_size - len(self.word2idx))
        for idx, (word, _) in enumerate(most_common, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def texts_to_sequences(self, texts):
        # Ensure that self.word2idx.get() does not lead to recursive calls
        return [
            [self.word2idx.get(word, self.word2idx["<OOV>"]) for word in text.split()]
            for text in texts
        ]


# 2. Dataset
class AuthorDataset(Sequence):
    def __init__(self, texts, authors, tokenizer, author_to_id, batch_size=BATCH_SIZE):
        self.texts = texts
        self.authors = authors
        self.tokenizer = tokenizer
        self.author_to_id = author_to_id
        self.batch_size = batch_size
        self.data = self.preprocess()

    def preprocess(self):
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        data = []
        for seq, author in zip(sequences, self.authors):
            for i in range(1, len(seq)):
                x = seq[:i]
                y = seq[i]
                data.append((x, y, self.author_to_id[author]))
        return data

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch, y_batch, author_batch = [], [], []
        for x, y, author_id in batch_data:
            x_padded = x + [0] * (MAX_SEQ_LEN - len(x)) if len(x) < MAX_SEQ_LEN else x[:MAX_SEQ_LEN]
            x_batch.append(x_padded)
            y_batch.append(y)
            author_batch.append(author_id)

        # Ensure np.array handles valid structures
        return [np.array(x_batch, dtype=np.int32), np.array(author_batch, dtype=np.int32)], np.array(y_batch, dtype=np.int32)

# Load data
data = pd.read_csv("~/data/author_data.csv")
texts = data["text"]
authors = data["author"]

# Tokenizer and author mapping
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
author_to_id = {author: idx for idx, author in enumerate(authors.unique())}

# Train/test split
train_texts, test_texts, train_authors, test_authors = train_test_split(
    texts, authors, test_size=0.2, random_state=42
)

# Create dataset
print('well')
train_dataset = AuthorDataset(train_texts, train_authors, tokenizer, author_to_id)
test_dataset = AuthorDataset(test_texts, test_authors, tokenizer, author_to_id)
print('well 2')

# 3. LSTM Model in TensorFlow
def build_model(vocab_size, author_count, embedding_dim, author_embedding_dim, hidden_dim):
    # Inputs
    text_input = Input(shape=(MAX_SEQ_LEN,))
    author_input = Input(shape=())

    # Embeddings
    word_embedding = Embedding(vocab_size, embedding_dim)(text_input)
    author_embedding = Embedding(author_count, author_embedding_dim)(author_input)
    author_embedding_repeated = tf.expand_dims(author_embedding, axis=1)
    author_embedding_repeated = tf.tile(author_embedding_repeated, [1, MAX_SEQ_LEN, 1])

    # Combine embeddings
    combined = Concatenate()([word_embedding, author_embedding_repeated])

    # LSTM Layer
    lstm_output = LSTM(hidden_dim, return_sequences=False)(combined)

    # Output Layer
    output = Dense(vocab_size, activation="softmax")(lstm_output)

    # Model
    model = Model(inputs=[text_input, author_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Initialize model
model = build_model(len(tokenizer.word2idx), len(author_to_id), EMBEDDING_DIM, AUTHOR_EMBEDDING_DIM, HIDDEN_DIM)
# Train the model
model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, verbose=1)
