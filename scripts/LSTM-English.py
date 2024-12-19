import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

nltk.download('stopwords')

# Load the data
data_path = 'TokenizedEnglishData/all_tokenized_data.csv'
data = pd.read_csv(data_path)

# Combine the text and author into a DataFrame
data_df = pd.DataFrame({'text': data['text'], 'author': data['author']})

# Find the minimum number of samples for any author
min_samples = data_df['author'].value_counts().min()

# Sample each author to have the same number of samples of each author in the train set
data_df_balanced = data_df.groupby('author', group_keys=False).apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Split the balanced data into features and labels
X_balanced = data_df_balanced['text']
y_balanced = data_df_balanced['author']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Remove stop words
stop_words = set(stopwords.words('english'))
X_train = X_train.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
X_test = X_test.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Tokenization and padding
max_words = 10000  # Limit on the number of words to use
max_sequence_length = 200  # Maximum length of the sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Convert labels to categorical
y_train_cat = pd.get_dummies(y_train).values
y_test_cat = pd.get_dummies(y_test).values

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(units=64, dropout=0.4, recurrent_dropout=0.4)))
model.add(Dense(units=y_train_cat.shape[1], activation='softmax', kernel_regularizer=l2(0.01)))  # Softmax for multi-class classification
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train_cat, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test_cat), verbose=1)

# Evaluate the model
y_pred_prob = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert y_test_cat to actual labels for evaluation
y_test_labels = np.argmax(y_test_cat, axis=1)

# Define a dictionary to map numerical labels to author names
label_to_author = {i: author for i, author in enumerate(y_train.unique())}

# Evaluate accuracy and f1 score
accuracy = accuracy_score(y_test_labels, y_pred)
f1 = f1_score(y_test_labels, y_pred, average='weighted')
report = classification_report(y_test_labels, y_pred, target_names=[label_to_author[i] for i in range(len(label_to_author))])

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print('Classification Report:')
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[label_to_author[i] for i in range(len(label_to_author))], yticklabels=[label_to_author[i] for i in range(len(label_to_author))])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Return the number of each author in the train set
author_counts_train = y_train.value_counts()
print("Number of samples for each author in the training set:")
print(author_counts_train)
