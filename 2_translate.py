import pandas as pd
import os
import time
from deep_translator import GoogleTranslator

# Inisialisasi translator
translator = GoogleTranslator(source='en', target='id')

def is_english_word(word):
    """Cek apakah sebuah kata valid dalam bahasa Inggris dengan mencoba menerjemahkannya ke bahasa Inggris."""
    translated = GoogleTranslator(source='auto', target='en').translate(word)
    return translated.lower() != word.lower()

def translate_text(text):
    """Menerjemahkan teks dari Inggris ke Indonesia hanya jika teks berbahasa Inggris."""
    words = text.split()
    translated_words = []
    
    for word in words:
        if is_english_word(word):  # Hanya translate jika kata terdeteksi bahasa Inggris
            try:
                translated_word = translator.translate(word)
                translated_words.append(translated_word)
            except Exception:
                translated_words.append(word)  # Jika gagal, gunakan kata asli
        else:
            translated_words.append(word)  # Jika bukan bahasa Inggris, tetap gunakan kata asli
    
    return ' '.join(translated_words)

# Path folder input dan output
input_folder = 'Output/'
output_folder = 'Output/'

# Daftar file yang akan diproses
input_files = [
    "2_Cleaned_data_ria_ricis.csv",
    "2_Cleaned_data_zaskia_sungkar.csv",
    "2_Cleaned_data_paula_verhoeven.csv",
    "2_Cleaned_data_citra_kirana.csv",
    "2_Cleaned_data_dian_pelangi.csv"
]

for file_name in input_files:
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name.replace("2_Cleaned_data_", "2_Translate_"))
    
    print(f"Mulai mentranslate file: {file_name}")
    start_time = time.time()
    df = pd.read_csv(input_path)
    
    if 'preprocessed_text' in df.columns:
        last_log_time = time.time()
        translated_texts = []
        
        for index, text in enumerate(df['preprocessed_text'].astype(str)):
            translated_texts.append(translate_text(text))
            
            # Logging setiap 15 detik
            if time.time() - last_log_time >= 15:
                print(f"Sudah memproses {index + 1} baris...")
                last_log_time = time.time()
        
        df['translated_text'] = translated_texts
        df[['translated_text']].to_csv(output_path, index=False)
        print(f"Translasi selesai untuk {file_name}. Hasil disimpan di {output_path}.")
    else:
        print(f"Kolom 'preprocessed_text' tidak ditemukan di {file_name}, proses dilewati.")
    
    end_time = time.time()
    print(f"Proses file {file_name} selesai dalam {end_time - start_time:.2f} detik.\n")
