import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. Download Required NLTK Resources ---
print("Downloading NLTK components...")
nltk.download('punkt')
nltk.download('punkt_tab')  # <-- ADD THIS LINE HERE
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
print("Downloads complete.\n" + "="*50)

stop_words = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)

# --- 2. Preprocessing Function ---
def preprocess_text(text):
    if not isinstance(text, str):
        return [], []
    cleaned_text = text.lower()
    cleaned_text = "".join([char for char in cleaned_text if char not in punctuation_set])
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    pos_tags = nltk.pos_tag(filtered_tokens)
    return filtered_tokens, pos_tags

# --- 3. Process Dataset 1: Clothing Reviews (50 Rows) ---
print("\nProcessing Dataset 1: Women's E-Commerce Clothing Reviews...")
# Column 'Review Text' contains the text, and it naturally has > 3 columns overall
df1 = pd.read_csv('clothing_reviews.csv')
df1_subset = df1.head(50).copy()

processed_results1 = df1_subset['Review Text'].apply(preprocess_text)
df1_subset['Tokens'] = [res[0] for res in processed_results1]
df1_subset['POS_Tags'] = [res[1] for res in processed_results1]

df1_subset.to_csv('cleaned_clothing_data.csv', index=False)
print("Saved 50 rows to 'cleaned_clothing_data.csv'")

# --- 4. Process Dataset 2: Twitter Sentiment (10,000 Rows) ---
print("\nProcessing Dataset 2: Twitter Sentiment...")
twitter_cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
df2 = pd.read_csv('twitter_data.csv', encoding='latin-1', names=twitter_cols)
df2_subset = df2.head(10000).copy()

processed_results2 = df2_subset['text'].apply(preprocess_text)
df2_subset['Tokens'] = [res[0] for res in processed_results2]
df2_subset['POS_Tags'] = [res[1] for res in processed_results2]

df2_subset.to_csv('cleaned_twitter_data.csv', index=False)
print("Saved 10,000 rows to 'cleaned_twitter_data.csv'")

# --- 5. Before & After Examples ---
print("\n" + "="*20 + " BEFORE & AFTER EXAMPLES " + "="*20)
print("\n--- Dataset 1 (Clothing Reviews) Example ---")
print(f"Original: {df1_subset['Review Text'].iloc[1]}")
print(f"Tokens:   {df1_subset['Tokens'].iloc[1]}")
print(f"POS Tags: {df1_subset['POS_Tags'].iloc[1]}")