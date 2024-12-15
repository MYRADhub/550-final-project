import re
import os
import glob
from unidecode import unidecode
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


french_stop_words = set(stopwords.words('french'))


# Used to clean the text by removing accents, numbers, symbols, and stop words
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


# Copy the content of the input file to the output file, cleaning the text in the process
def copy_and_clean_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        cleaned_text = clean_text(text)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
    except Exception as e:
        print(f"An error occurred: {e}")


# Process all .txt files in the directory and its subdirectories
def process_all_files(directory):
    txt_files = glob.glob(os.path.join(directory, '**', '*.txt'), recursive=True)
    
    for input_file_path in txt_files:
        # Create a new file for the cleaned text
        output_file_path = input_file_path.replace('.txt', '_cleaned.txt')
        copy_and_clean_file(input_file_path, output_file_path)


directory = 'FrenchData'
process_all_files(directory)