import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the data
data_path = 'TokenizedData/all_tokenized_data.csv'
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
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Ensure there is no data leakage by comparing indices
train_indices = X_train.index
test_indices = X_test.index
leakage = set(train_indices).intersection(set(test_indices))
if leakage:
    print(f"Data leakage detected in indices: {leakage}")
else:
    print("No data leakage detected.")


french_stop_words = stopwords.words('french')

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words=french_stop_words)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# The Naive Bayes classifier
nb_classifier = MultinomialNB()

# The hyperparameter grid for GridSearchCV
param_grid = {
    'alpha': [0.0001, 0.001, 0.1, 0.5, 1.0, 2.0, 5.0] 
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)

# Perform Grid Search
grid_search.fit(X_train_tfidf, y_train)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print('Classification Report:')
print(report)

# Return the number of each author in the train set
author_counts_train = y_train.value_counts()
print("Number of samples for each author in the training set:")
print(author_counts_train)
