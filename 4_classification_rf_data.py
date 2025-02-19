import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Gabungkan semua file CSV menjadi satu DataFrame
file_paths = [
    "Output/3_lexicon_2_Cleaned_1_Filter_data_ria_ricis.csv",
    "Output/3_lexicon_2_Cleaned_1_Filter_data_zaskia_sungkar.csv",
    "Output/3_lexicon_2_Cleaned_1_Filter_data_paula_verhoeven.csv",
    "Output/3_lexicon_2_Cleaned_1_Filter_data_citra_kirana.csv",
    "Output/3_lexicon_2_Cleaned_1_Filter_data_dian_pelangi.csv"
]

print("Menggabungkan semua file CSV...")
dataframes = [pd.read_csv(file_path) for file_path in file_paths]
data = pd.concat(dataframes, ignore_index=True)

# Log jumlah total data
print(f"Total data sebelum filter: {len(data)} baris")

# Pastikan kolom yang diperlukan ada
if 'preprocessed_text' not in data.columns or 'label_sentiment' not in data.columns:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment' tidak ditemukan dalam CSV!")

# Hapus data dengan label 'netral' (jika ada)
print("Menghapus data dengan label 'netral'...")
data = data[data['label_sentiment'] != 'netral']

# Map label sentimen ke angka
label_mapping = {'positif': 1, 'negatif': 0}
data['label_sentiment_number'] = data['label_sentiment'].map(label_mapping)

# Log jumlah data setelah filter
print(f"Total data setelah filter: {len(data)} baris")

# Ambil teks dan label
texts = data['preprocessed_text'].astype(str)
labels = data['label_sentiment_number'].astype(int)

# Vektorisasi TF-IDF
print("Melakukan vektorisasi TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Split data menjadi training dan testing
print("Membagi data menjadi training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Log ukuran data setelah split
print(f"Data Training: {X_train.shape[0]} baris, Data Testing: {X_test.shape[0]} baris")

# Hyperparameter tuning untuk Random Forest
print("\nRandom Forest Classifier")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inisialisasi RandomizedSearchCV dengan verbose untuk log progres
rf_grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2  # Log progres pencarian parameter terbaik
)

# Proses training dengan log progres
print("Memulai pelatihan model...")
rf_grid.fit(X_train, y_train)
print("Pelatihan model selesai!")

# Model terbaik
best_rf = rf_grid.best_estimator_

# Prediksi pada data testing
print("Melakukan prediksi pada data testing...")
y_pred_rf = best_rf.predict(X_test)

# Evaluasi
print("Evaluasi model...")
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Best Parameters for Random Forest: {rf_grid.best_params_}")
print("Akurasi:", rf_accuracy)
print(classification_report(y_test, y_pred_rf, labels=[0, 1], target_names=["Negatif", "Positif"], zero_division=0))

# Simpan hasil evaluasi
evaluation_results = pd.DataFrame([{
    'Model': 'Random Forest Classifier',
    'Accuracy': rf_accuracy,
    'Best Parameters': rf_grid.best_params_
}])

output_eval_file = 'Output/4_Classification_random_forest.csv'
evaluation_results.to_csv(output_eval_file, index=False)

# Log penyimpanan hasil
print("\nHasil Evaluasi Model:")
print(evaluation_results)
print(f"Hasil evaluasi akurasi telah disimpan ke {output_eval_file}")
