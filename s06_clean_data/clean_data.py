import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# # Download the stopwords from nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# Set path of nltk_data
script_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_path = os.path.join(script_dir, "nltk_data")
nltk.data.path.append(nltk_data_path)

DATA_FOLDER = 'data'

def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

def to_lowercase(sentence):
    return sentence.lower()

stop_words = set(stopwords.words('english'))
def remove_stopwords(sentence):
    words = word_tokenize(sentence)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def stem_or_lemmatize(sentence, method='stem'):
    if method == 'stem':
        stemmer = PorterStemmer()
        words = word_tokenize(sentence)
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    elif method == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(sentence)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    else:
        raise ValueError("method 参数必须是 'stem' 或 'lemmatize'")

def clean_sentence(sentence):
    sentence = remove_punctuation(sentence)
    sentence = to_lowercase(sentence)
    sentence = remove_stopwords(sentence)
    sentence = stem_or_lemmatize(sentence)
    return sentence

def remove_duplicates(data):
    data_deduplicated = data.drop_duplicates(subset=['utterance', 'intent'])
    return data_deduplicated

def remove_empty_utterances(data):
    data_cleaned = data[data['utterance'].str.strip().astype(bool)]
    return data_cleaned

def main():
    # Read input csv
    parent_dir = os.path.dirname(script_dir)
    input_file = os.path.join(parent_dir, "s05_random_swap", DATA_FOLDER, "data_after_swap.csv")
    df = pd.read_csv(input_file)

    df_to_clean = df

    # Process each sentence
    cleaned_data = []
    for sentence, intent in zip(df_to_clean["utterance"], df_to_clean["intent"]):
        cleaned_sentences = clean_sentence(sentence)
        cleaned_data.append({"utterance": cleaned_sentences, "intent": intent})

    df_cleaned = pd.DataFrame(cleaned_data)
    df_cleaned = remove_duplicates(df_cleaned)
    df_cleaned = remove_empty_utterances(df_cleaned)

    # Export
    output_cleaned = os.path.join(script_dir, DATA_FOLDER, "data_after_clean.csv")
    df_cleaned.to_csv(output_cleaned, index=False)
    print(f"Cleaned data has been saved")

if __name__ == "__main__":
    main()
