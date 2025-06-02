import pandas as pd
import numpy as np  # Import numpy for NaN handling
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv("imdb_top_1000.csv")

# Choose columns you want to use
columns_to_use = ['Series_Title', 'Overview', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4','No_of_Votes','Gross']

# Clean your columns
def clean_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Convert text to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters and extra whitespaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        # Join tokens back into string
        cleaned_text = ' '.join(lemmatized_tokens)
        return cleaned_text
    else:
        return ''  # Return empty string for NaN values



# Concatenate the columns
df['concatenated_text'] = df[columns_to_use].apply(lambda x: ' '.join(x), axis=1)



# Save the cleaned and concatenated data
df.to_csv("cleaned_imdb_data.csv", index=False)
