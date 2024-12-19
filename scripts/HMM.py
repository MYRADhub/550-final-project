import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, f1_score
from hmmlearn.hmm import GaussianHMM
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from gensim.models import Word2Vec

# French stop words
nltk.download('stopwords')
nltk.download('punkt')
french_stop_words = stopwords.words('french') # Change this depending on language used

# Load the data
data_path = 'TokenizedData/all_tokenized_data.csv' #Change to the path of the tokenized data
data = pd.read_csv(data_path)

# Combine the text and author into a DataFrame
data_df = pd.DataFrame({'text': data['text'], 'author': data['author']})

# Find the minimum number of samples for any author
min_samples = data_df['author'].value_counts().min()

# Undersample each author to have the same number of samples of each author in the train set
data_df_balanced = data_df.groupby('author', group_keys=False).apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Split the balanced data into features and labels
X_balanced = data_df_balanced['text']
y_balanced = data_df_balanced['author']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

# Tokenize and create sequences using Word2Vec
def tokenize_and_embed(texts):
    tokenized_texts = [word_tokenize(text) for text in texts]
    model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    sequences = [np.array([model.wv[word] for word in text if word in model.wv], dtype=np.float32) for text in tokenized_texts]
    return sequences

X_train_seq = tokenize_and_embed(X_train)
X_test_seq = tokenize_and_embed(X_test)

# Group data by author for HMM training
train_data_by_author = {author: [] for author in y_train.unique()}
for text, label in zip(X_train_seq, y_train):
    train_data_by_author[label].append(text)

# Hyperparameter tuning: Experimenting with n_components for HMM
models = {}
param_grid = {
    'n_components': [3, 5, 7, 10, 15, 20, 25],
    'n_iter': [50, 100],
    'tol': [1e-2, 1e-3]
}
for author, sequences in train_data_by_author.items():
    lengths = [len(seq) for seq in sequences]
    concatenated = np.concatenate(sequences)
    best_score = float('-inf')
    best_params = None
    
    # Grid search over hyperparameters
    for params in ParameterGrid(param_grid):
        hmm_model = GaussianHMM(
            n_components=params['n_components'],
            n_iter=params['n_iter'],
            tol=params['tol'],
            random_state=42
        )
        try:
            hmm_model.fit(concatenated, lengths)
            score = hmm_model.score(concatenated)
            if score > best_score:
                best_score = score
                best_params = params
        except:
            continue
    
    # Train the model with the best hyperparameters
    if best_params:
        hmm_model = GaussianHMM(
            n_components=best_params['n_components'],
            n_iter=best_params['n_iter'],
            tol=best_params['tol'],
            random_state=42
        )
        hmm_model.fit(concatenated, lengths)
        models[author] = hmm_model

# Evaluate on the test set
y_pred = []
for seq in X_test_seq:
    best_score = float('-inf')
    best_author = None
    for author, model in models.items():
        try:
            score = model.score(seq)
            if score > best_score:
                best_score = score
                best_author = author
        except:
            continue
    y_pred.append(best_author)

# Evaluate the predictions
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print('Classification Report:')
print(report)

# Print a few samples from the test set with their predicted and actual labels
print("\nSample predictions vs actual labels:")
for i in range(min(10, len(y_test))):  # Print up to 10 samples
    print(f"Sample {i+1}: Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# Return the number of each author in the train set
author_counts_train = y_train.value_counts()
print("Number of samples for each author in the training set:")
print(author_counts_train)