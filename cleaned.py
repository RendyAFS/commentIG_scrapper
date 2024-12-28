import pandas as pd
import os
import json
import re
import emoji

# Load slank word dictionary
def load_slank_word_dictionary(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        slank_dict = json.load(file)
    return slank_dict

# Remove emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

# Remove URLs from text
def remove_urls(text):
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, "", text)

# Remove punctuation from text
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Clean the content text by replacing slang words with standard words
def clean_text(text, slank_dict):
    # Remove emojis
    text = remove_emojis(text)
    
    # Remove URLs
    text = remove_urls(text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Replace slang/typo words with standard words
    words = text.split()
    filtered_words = [slank_dict.get(word.lower(), word) for word in words]
    return " ".join(filtered_words)

# Load dataset from local CSV file
def load_dataset(file_path):
    return pd.read_csv(file_path)

def main():
    # Directories
    dataset_dir = "./Dataset"
    output_dir = "./Output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load slang dictionary
    slank_dict_path = os.path.join(dataset_dir, "slank_word_dictionary.txt")
    slank_dict = load_slank_word_dictionary(slank_dict_path)

    # List of dataset files
    dataset_files = [
        "data_ria_ricis.csv",
        "data_zaskia_sungkar.csv",
        "data_paula_verhoeven.csv",
        "data_citra_kirana.csv",
        "data_dian_pelangi.csv",
    ]

    # Process each dataset file
    for file_name in dataset_files:
        dataset_path = os.path.join(dataset_dir, file_name)
        df = load_dataset(dataset_path)

        # Check if the "Content" column exists and clean it
        if "Content" in df.columns:
            df["Content"] = df["Content"].apply(lambda x: clean_text(str(x), slank_dict))

            # Save the cleaned data with only the "Content" column
            output_file_name = f"Cleaned_data_{file_name}"
            output_file_path = os.path.join(output_dir, output_file_name)
            df[["Content"]].to_csv(output_file_path, index=False, encoding="utf-8")
            print(f"Saved cleaned data to {output_file_path}")

if __name__ == "__main__":
    main()
