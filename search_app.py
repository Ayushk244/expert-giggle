"""
    Create an Streamlit app that does the following:

    - Reads an input from the user
    - Embeds the input
    - Search the vector DB for the entries closest to the user input
    - Outputs/displays the closest entries found
"""
import streamlit as st
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


ssl_context = create_default_context()
# Store embeddings in Elasticsearch
ssl_context.check_hostname = False
ssl_context.verify_mode = CERT_NONE
model = SentenceTransformer('paraphrase-distilroberta-base-v2')

es = Elasticsearch(['https://localhost:9200'], basic_auth=("elastic","gq4gBfcTSJvPs4opA0zt"),ssl_context=ssl_context)
if(es.ping):
    print("Elastic search  working")
else:
    print("Elastic search not working")

# Check if the index exists
index_name = 'imdb_embeddings'
index_exists = es.indices.exists(index=index_name)
if index_exists:
    print(f"The index '{index_name}' exists.")
else:
    print(f"The index '{index_name}' does not exist.")


model = SentenceTransformer('paraphrase-distilroberta-base-v2')

def embed_text(text):
    # Embed the text using Sentence Transformer
    embedding = model.encode(text)
    return embedding.tolist()

def search_similar_entries(user_input, top_n=1):
    # Embed the user input
    user_embedding = embed_text(user_input)
    
    # Construct Elasticsearch query for semantic search
    query = {
    "knn": {
        "field": "embedding",  # Specify the field containing the embeddings
        "query_vector": user_embedding,
        "k": 10  # Number of nearest neighbors to retrieve
    }
}
    
    # Perform the search
    res = es.search(index='imdb_embeddings', body=query)
    
    return res['hits']['hits']

def main():
    st.title("Semantic Search App")
    
    # Get user input
    user_input = st.text_input("Enter your query:")
    
    if st.button("Search"):
        if user_input:
            # Perform semantic search
            results = search_similar_entries(user_input)
            
            # Display results
            st.write("Top Similar Entries:")
            for idx, result in enumerate(results, 1):
                # st.write(f"{idx}. Title: {result['_source']['Series_Title']},Release Year: {result['_source']['Released_Year']},Genre: {result['_source']['Genre']}, Director: {result['_source']['Director']},IMDB Rating: {result['_source']['IMDB_Rating']},Director: {result['_source']['Director']},Overview: {result['_source']['Overview']}, Meta score: {result['_source']['Meta_score']}")
                # st.write(result['_source'])
                image_url = result['_source']['Poster_Link']
                st.image(image_url, caption='Image')
                Title = result['_source']['Series_Title']

                capitalized_Title = Title.capitalize()

                st.subheader(capitalized_Title)

                st.subheader(result['_source']['Released_Year'])
                overview = result['_source']['Overview']
                capitalized_overview = overview.capitalize()

                st.write(capitalized_overview)
                #st.write(result['_source']['Overview'])
                st.write("Genre:", result['_source']['Genre'])
                st.write("Director:", result['_source']['Director'])


if __name__ == "__main__":
    main()