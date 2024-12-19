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
data_path = 'TokenizedFrenchData/all_tokenized_data.csv'
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

# Define stop words
french_stop_words = stopwords.words('french')

# Feature extraction using TF-IDF with adjusted parameters
vectorizer = TfidfVectorizer(
    stop_words=french_stop_words,
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
sample_text = "Dans les profondeurs de l'océan Atlantique, là où les vagues s'écrasent contre les récifs invisibles, un voyage extraordinaire était sur le point de débuter. Le capitaine Armand Delacour, un explorateur intrépide aux cheveux argentés, se tenait sur le pont de son navire, le Vingtième Siècle, contemplant l'immensité bleue. Son regard perçant scrutait l'horizon, à la recherche de l'inconnu, car c'était sa destinée de braver les mystères des mers."
predicted_author = predict_author(sample_text)
print(f"The predicted author for the given text is: {predicted_author}")


import numpy as np

# Function to generate text based on a target author
def generate_text(model, vectorizer, target_author, max_len=50, temperature=0.7):
    """
    Generates text in the style of the specified author.
    
    :param model: Trained Naive Bayes model
    :param vectorizer: Trained TF-IDF vectorizer
    :param target_author: The author whose style should be mimicked
    :param max_len: Maximum length of the generated text
    :param temperature: Temperature parameter to control diversity (lower for less diversity)
    :return: Generated text
    """
    if target_author not in model.classes_:
        raise ValueError(f"Author '{target_author}' not found in the model classes.")
    
    # Seed with an initial word or start of sentence
    seed_text = [""]  # Start with empty text
    current_text = seed_text[0]
    
    # Generate text iteratively
    for _ in range(max_len):
        # Generate candidate words by sampling from the feature space
        feature_names = vectorizer.get_feature_names_out()
        author_probs = model.feature_log_prob_[model.classes_ == target_author]
        
        # Normalize probabilities with temperature scaling
        scaled_probs = np.exp(author_probs / temperature)
        scaled_probs /= scaled_probs.sum()
        
        # Sample next word
        next_word_idx = np.random.choice(len(feature_names), p=scaled_probs[0])
        next_word = feature_names[next_word_idx]
        
        # Append word to text
        current_text += " " + next_word
        
        # Update seed_text
        seed_text = [current_text]
    
    return current_text.strip()

# Example usage
target_author = "Voltaire"  # Replace with an actual author name from your dataset
seed_word = "géomètre"  # Optional seed word

# Print available classes
print("Available classes:", best_model.classes_)

# Check if target_author is in the available classes
if target_author not in best_model.classes_:
    raise ValueError(f"Author '{target_author}' not found in the model classes. Available classes are: {best_model.classes_}")

generated_text = generate_text(best_model, vectorizer, target_author, max_len=50, temperature=0.7)
print(f"Generated text in the style of {target_author}:\n{generated_text}")

