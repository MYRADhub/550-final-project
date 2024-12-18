import os
import glob
import spacy

nlp = spacy.load('fr_core_news_sm')
nlp.max_length = 1500000

# Function to clean and tokenize the french texts
def clean_and_tokenize_text(text):
    # Process the text with SpaCy
    doc = nlp(text.lower())

    # Filter tokens: remove stop words, punctuation, and keep only alphabetic tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    return ' '.join(tokens)

# Function to process all files in a directory and merge the tokenized texts for each author into a single file
def process_files(directory, tokenized_output_directory):
    # Recursively find all txt files in the directory and its subdirectories
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

    # Write the merged texts for each author
    for author_name, texts in author_texts.items():
        merged_text = ' '.join(texts)
        author_output_file_path = os.path.join(tokenized_output_directory, f"{author_name}_tokenized.txt")
        
        # Ensure the output directories exist
        os.makedirs(os.path.dirname(author_output_file_path), exist_ok=True)
        
        with open(author_output_file_path, 'w', encoding='utf-8') as file:
            file.write(merged_text)

directory = 'FrenchData' # The directory containing the French texts
tokenized_output_directory = 'TokenizedFrenchData' # The directory where the tokenized texts will be saved
process_files(directory, tokenized_output_directory)