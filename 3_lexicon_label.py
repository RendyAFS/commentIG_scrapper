import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import os
import time

# Unduh lexicon VADER
nltk.download('vader_lexicon')

# Path ke file lexicon bahasa Indonesia
path = 'Dataset/sentiwords_id.txt'  # Pastikan file ini sudah ada
df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])

# Buat dictionary dari lexicon bahasa Indonesia
senti_dict = {row['word']: float(row['value']) for _, row in df_senti.iterrows()}

# Inisialisasi Stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi SentimentIntensityAnalyzer dan update lexicon-nya
senti_indo = SentimentIntensityAnalyzer()
senti_indo.lexicon.update(senti_dict)

# Daftar file yang akan diproses
files = [
    "Output/2_Cleaned_data_ria_ricis.csv",
    "Output/2_Cleaned_data_zaskia_sungkar.csv",
    "Output/2_Cleaned_data_paula_verhoeven.csv",
    "Output/2_Cleaned_data_citra_kirana.csv",
    "Output/2_Cleaned_data_dian_pelangi.csv"
]

# Fungsi untuk tokenisasi menggunakan regex
def tokenize(text):
    return re.findall(r'\w+', text)

# Iterasi melalui setiap file dalam daftar
for file in files:
    input_path = file
    output_filename = "Output/3_Lexicon_label_" + os.path.basename(file).replace("2_Cleaned_data_", "")
    
    print(f"Mulai memproses file: {file}")
    start_time = time.time()

    # Load data dari file CSV
    df = pd.read_csv(input_path)

    # Pastikan kolom 'preprocessed_text' ada dan adalah string
    if 'preprocessed_text' not in df.columns:
        print(f"Kolom 'preprocessed_text' tidak ditemukan di {file}, proses dilewati.")
        continue
    
    df['preprocessed_text'] = df['preprocessed_text'].astype(str)

    # List untuk menyimpan hasil sentimen
    label_lexicon = []

    # Iterasi melalui setiap baris di DataFrame
    for index, row in df.iterrows():
        # Stemming pada teks
        stemmed_text = stemmer.stem(row['preprocessed_text'])
        
        # Hitung skor sentimen untuk teks yang sudah distem
        score = senti_indo.polarity_scores(stemmed_text)
        
        # Tentukan label sentimen hanya dengan kata positif dan negatif
        if score['compound'] >= 0.05:
            label_lexicon.append("positif")  # positif
        elif score['compound'] <= -0.05:
            label_lexicon.append("negatif")  # negatif
        else:
            # If no condition is met, append "negatif" as a default value
            label_lexicon.append("negatif")

        # Log every 1000 processed rows or last row
        if (index + 1) % 1000 == 0 or index == len(df) - 1:
            print(f"Proses: {index + 1}/{len(df)} baris selesai...")

    # Ensure label_lexicon has the same length as the DataFrame
    if len(label_lexicon) != len(df):
        print(f"Warning: jumlah label ({len(label_lexicon)}) tidak cocok dengan jumlah baris DataFrame ({len(df)}).")

    # Tambahkan hasil sentimen sebagai kolom baru di DataFrame
    df['label_sentiment'] = label_lexicon

    # Simpan DataFrame yang sudah diberi label ke file CSV baru dengan prefix '3_lexicon_'
    df.to_csv(output_filename, index=False)

    print(f"Proses selesai untuk {file}. Hasil sentimen telah disimpan di {output_filename}.")
    
    print(df['label_sentiment'].value_counts())

    end_time = time.time()
    print(f"Proses file {file} selesai dalam {end_time - start_time:.2f} detik.\n")
