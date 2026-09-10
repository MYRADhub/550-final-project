import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the data
data_path = 'TokenizedEnglishData/all_tokenized_data.csv'
data = pd.read_csv(data_path)

# Combine the text and author into a DataFrame
data_df = pd.DataFrame({'text': data['text'], 'author': data['author']})

# Split the original data into train and test sets with stratified sampling
X = data_df['text']
y = data_df['author']
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Find the minimum number of samples for any author in the training set
min_samples = y_train_orig.value_counts().min()

# Balance the training set
train_df_balanced = pd.DataFrame({'text': X_train_orig, 'author': y_train_orig}).groupby('author', group_keys=False).apply(
    lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Extract balanced train data
X_train_balanced = train_df_balanced['text']
y_train_balanced = train_df_balanced['author']

# Define stop words for English
english_stop_words = stopwords.words('english')

# Feature extraction using TF-IDF with adjusted parameters
vectorizer = TfidfVectorizer(
    stop_words=english_stop_words,  # Change to English stop words
    min_df=5,  # Ignore very rare words
    max_df=0.8,  # Ignore overly frequent words
    max_features=10000,  # Limit feature set size
    ngram_range=(1, 2)  # Include unigrams and bigrams
)
X_train_tfidf = vectorizer.fit_transform(X_train_balanced)
X_test_tfidf = vectorizer.transform(X_test_orig)

# The Naive Bayes classifier
nb_classifier = MultinomialNB()

# The hyperparameter grid for GridSearchCV
param_grid = {
    'alpha': [0.0001, 0.001, 0.1, 0.5, 1.0, 2.0, 5.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)

# Perform Grid Search
grid_search.fit(X_train_tfidf, y_train_balanced)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Evaluate the best model on the original test set
y_pred = best_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test_orig, y_pred)
f1 = f1_score(y_test_orig, y_pred, average='weighted')
report = classification_report(y_test_orig, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print('Classification Report:')
print(report)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test_orig, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Return the number of each author in the train set
author_counts_train = y_train_balanced.value_counts()
print("Number of samples for each author in the balanced training set:")
print(author_counts_train)

def predict_author(text):
    # Transform the input text using the trained TF-IDF vectorizer
    text_tfidf = vectorizer.transform([text])
    
    # Predict the author using the trained model
    prediction = best_model.predict(text_tfidf)
    
    return prediction[0]

# Example usage
sample_text = "In the deep waters of the Atlantic Ocean, where waves crash against invisible reefs, an extraordinary journey was about to begin. Captain Armand Delacour, a fearless explorer with silver hair, stood on the deck of his ship, the Twentieth Century, gazing at the endless blue. His sharp eyes scanned the horizon, searching for the unknown, for it was his destiny to brave the mysteries of the seas."
predicted_author = predict_author(sample_text)
print(f"The predicted author for the given text is: {predicted_author}")
