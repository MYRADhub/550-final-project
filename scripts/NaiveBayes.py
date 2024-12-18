import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
import nltk
from nltk.corpus import stopwords

# Load the data
data_path = 'TokenizedFrenchData/all_tokenized_data.csv'
data = pd.read_csv(data_path)

# Combine the text and author into a DataFrame
data_df = pd.DataFrame({'text': data['text'], 'author': data['author']})

# Find the minimum number of samples for any author
min_samples = data_df['author'].value_counts().min()

# Undersample each author to have the same number of samples
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

# Download and use French stop words from nltk
nltk.download('stopwords')
french_stop_words = stopwords.words('french')

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words=french_stop_words)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test_tfidf)

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