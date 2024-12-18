import os
import glob
import spacy
import csv
from unidecode import unidecode
nlp = spacy.load('fr_core_news_sm')
nlp.max_length = 1500000


def clean_and_tokenize_text(text):
    # Process the text with SpaCy
    doc = nlp(text.lower())

    # Filter tokens: remove stop words, punctuation, and keep only alphabetic tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # Remove accents and special characters
    cleaned_tokens = [unidecode(token) for token in tokens]

    return ' '.join(cleaned_tokens)



def split_into_chunks(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])


# Function to process all files in a directory and merge the tokenized texts for each author into a single file
def process_files(directory, tokenized_output_directory):
    # Recursively find all .txt files in the directory and its subdirectories
    txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)

    if not os.path.exists(tokenized_output_directory):
        os.makedirs(tokenized_output_directory)

    author_texts = {}

    for input_file_path in txt_files:
        try:
            print(f"Starting to process file: {input_file_path}")
            with open(input_file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            cleaned_tokenized_text = clean_and_tokenize_text(text)

            # Get the author name from the relative path
            relative_path = os.path.relpath(input_file_path, directory)
            author_name = os.path.dirname(relative_path)

            if author_name not in author_texts:
                author_texts[author_name] = []

            author_texts[author_name].append(cleaned_tokenized_text)

            print(f"Finished processing file: {input_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Write the chunks for each author into a single CSV file
    output_file_path = os.path.join(tokenized_output_directory, "all_tokenized_data.csv")

    with open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['author', 'text'])  # Write header
        for author_name, texts in author_texts.items():
            merged_text = ' '.join(texts)
            for chunk in split_into_chunks(merged_text):
                writer.writerow([author_name, chunk])

    print("All files have been processed and the data has been written to all_tokenized_data.csv")

directory = 'FrenchData'
tokenized_output_directory = 'TokenizedFrenchData'
process_files(directory, tokenized_output_directory)