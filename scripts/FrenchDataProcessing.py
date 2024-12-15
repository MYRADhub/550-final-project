import re
import os
import glob
from unidecode import unidecode
from nltk.corpus import stopwords

# Ensure you have the French stopwords downloaded
import nltk
nltk.download('stopwords')

# Get the list of French stop words
french_stop_words = set(stopwords.words('french'))

def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove accents
    text = unidecode(text)
    # Remove form feed characters
    text = text.replace('\f', '')
    # Remove all numbers and specific symbols, but keep letters including accented ones
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stop words
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word not in french_stop_words)
    return cleaned_text

def copy_and_clean_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        cleaned_text = clean_text(text)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
    except Exception as e:
        print(f"An error occurred: {e}")

def process_all_files_in_directory(directory):
    # Recursively find all .txt files in the directory and its subdirectories
    txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)
    
    for input_file_path in txt_files:
        # Create the output file path by appending '_cleaned' before the file extension
        output_file_path = input_file_path.replace('.txt', '_cleaned.txt')
        copy_and_clean_file(input_file_path, output_file_path)

# Example usage
directory = 'FrenchData'
process_all_files_in_directory(directory)