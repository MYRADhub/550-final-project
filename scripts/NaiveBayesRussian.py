from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import random
import numpy as np

def load_and_preprocess(data_path):
    """
    Load and preprocess dataset for Naive Bayes.
    Args:
        data_path (str): Path to the dataset CSV file.
    Returns:
        pd.DataFrame: Preprocessed dataset with 'text' and 'author' columns.
    """
    data = pd.read_csv(data_path)
    return data

# Prepare data
data_path = "../data/Russian/all_tokenized_data.csv"  # Replace with your dataset
data = load_and_preprocess(data_path)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['author'], test_size=0.2, random_state=42
)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Evaluate the model
accuracy = nb_model.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy:.2f}")

def generate_text(nb_model, vectorizer, author, max_len=50, seed_word=None, temperature=1.0):
    """
    Generate text in the style of a given author using a trained Naive Bayes model with sampling.
    Args:
        nb_model: Trained Naive Bayes model.
        vectorizer: Fitted CountVectorizer.
        author (str): Author whose style to mimic.
        max_len (int): Maximum length of the generated text.
        seed_word (str): Optional starting word.
        temperature (float): Sampling temperature to control randomness.
    Returns:
        str: Generated text.
    """
    if seed_word is None:
        # Randomly choose a starting word from the vocabulary
        seed_word = random.choice(vectorizer.get_feature_names_out())

    generated_text = [seed_word]
    current_word = seed_word

    for _ in range(max_len - 1):
        # Create a pseudo-document for the current word
        pseudo_doc = " ".join(generated_text)

        # Vectorize the pseudo-document
        vec = vectorizer.transform([pseudo_doc])

        # Get probabilities for the next word
        author_index = np.where(nb_model.classes_ == author)[0][0]
        word_probs = np.exp(nb_model.feature_log_prob_[author_index])  # Convert log probs to probabilities

        # Adjust probabilities with temperature
        word_probs = word_probs ** (1 / temperature)
        word_probs /= np.sum(word_probs)  # Normalize probabilities

        # Sample the next word based on probabilities
        next_word_idx = np.random.choice(len(word_probs), p=word_probs)
        next_word = vectorizer.get_feature_names_out()[next_word_idx]

        generated_text.append(next_word)
        current_word = next_word

    return " ".join(generated_text)

# Example usage
target_author = "bulgakov "  # Replace with an actual author name from your dataset, to see them all run print(nb_model.classes_)
seed_word = "даже"  # Optional seed word
generated_text = generate_text(nb_model, vectorizer, target_author, max_len=50, temperature=0.7)
print(f"Generated text in the style of {target_author}:\n{generated_text}")
