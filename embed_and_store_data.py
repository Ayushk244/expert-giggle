"""
- Prepare the text to embed for each reccord of your dataset.
    - Create the reccord.
        - Clean the text.
        - Concatenate fields.
- Choose a Sentence Embedding Model.
- Embed the text generated in the previous step for each reccord.
- Store the embeddings in a vector database (i.e. elasticsearch).
"""
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from ssl import create_default_context, CERT_NONE
import numpy as np


# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv("cleaned_imdb_data_with_nan_check.csv")


# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v2')


# Store embeddings in Elasticsearch



es = Elasticsearch(['https://localhost:9200'])
if(es.ping):
    print("Elastic search  working")
else:
    print("Elastic search not working")
transport_options = {"ignore": [400, 404]}
es.indices.delete(index='embeddings', **transport_options)


def embed_and_store(row):
    # Embed text from selected columns
    row = row.astype(str)
    
    embedding = model.encode(row['concatenated_text'])
    series_title = row['Series_Title']

    if isinstance(row['Series_Title'], str):
        
        series_title = series_title.strip()
    if(series_title!=""  ):
        if(pd.isna(series_title)==False):
    # Prepare document to store in Elasticsearch
            doc = {
              
                'concatenated_text': row['concatenated_text'],  # Store concatenated text for reference
                'embedding': embedding.tolist()  # Store embedding as a list
            }
            
            # Index document in Elasticsearch
            es.index(index='embeddings', body=doc)

# Iterate over each row in the DataFrame and embed/store the data
for _, row in df.iterrows():
    embed_and_store(row)



# Refresh index to make changes visible
es.indices.refresh(index='embeddings')
