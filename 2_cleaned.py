import pandas as pd
import re
import string
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import time
import emoji  # Import pustaka emoji

# Membuat stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def simple_tokenize(text):
    return text.split()

def normalize_slang(text, slang_dict):
    words = simple_tokenize(text)
    normalized_words = [slang_dict.get(re.sub(r'(.)\1+', r'\1\1', word), word) for word in words]
    return ' '.join(normalized_words)

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def preprocess_text(text, slang_dict, stemmer):
    text = text.lower()
    text = re.sub(r'@\w+\s*', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'[‘’“”…♪♪]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'b ', ' ', text)
    text = re.sub(r'rt ', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = remove_emoji(text).strip()
    if not text:
        return ''
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'\b([aiueo])\1{2,}\b', '', text)
    text = re.sub(r'\b(wkwk+|hahaha+|hehee+|hehehe+|wkw+|ahahaha+|lol+|rofl+|lmao+|lmfao+)\b', '', text)
    tokens = simple_tokenize(text)
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [slang_dict.get(re.sub(r'(.)\1+', r'\1\1', word), word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens).strip()

def verify_cleaned_text(text, slang_dict):
    return normalize_slang(text, slang_dict)

with open('Dataset/slank_word_dictionary.txt', 'r') as file:
    slang_dict = json.load(file)
    slang_dict = {re.sub(r'(.)\1+', r'\1\1', key): value for key, value in slang_dict.items()}

input_files = [
    "1_Filter_data_ria_ricis.csv",
    "1_Filter_data_zaskia_sungkar.csv",
    "1_Filter_data_paula_verhoeven.csv",
    "1_Filter_data_citra_kirana.csv",
    "1_Filter_data_dian_pelangi.csv"
]

input_folder = 'Output/'
output_folder = 'Output/'

for file_name in input_files:
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name.replace("1_Filter_data_", "2_Cleaned_data_"))
    
    print(f"Mulai memproses file: {file_name}")
    start_time = time.time()
    df = pd.read_csv(input_path)
    
    if 'Content' in df.columns:
        print(f"File {file_name} memiliki {len(df)} baris data.")
        processed_texts = []
        last_log_time = time.time()
        
        for index, row in df.iterrows():
            cleaned_text = preprocess_text(str(row['Content']), slang_dict, stemmer)
            cleaned_text = verify_cleaned_text(cleaned_text, slang_dict)
            processed_texts.append(cleaned_text)
            
            if time.time() - last_log_time >= 15:
                print(f"Proses: {index + 1}/{len(df)} baris selesai...")
                last_log_time = time.time()
        
        df['preprocessed_text'] = processed_texts
        df[['preprocessed_text']].to_csv(output_path, index=False)
        print(f"Preprocessing selesai untuk {file_name}. Hasil disimpan di {output_path}.")
    else:
        print(f"Kolom 'Content' tidak ditemukan di {file_name}, proses dilewati.")
    
    end_time = time.time()
    print(f"Proses file {file_name} selesai dalam {end_time - start_time:.2f} detik.\n")
