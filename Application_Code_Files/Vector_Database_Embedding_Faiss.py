# Import necessary libraries
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the dataset from a CSV file into a Pandas DataFrame
# Replace the file path with the actual location of your dataset
df = pd.read_csv('E:\\IMDB_MOVIES_PLATFORM\\Dataset\\Sample_IMDB_Movie_Dataset.csv')
# Combine all columns in each row into a single text string for embedding purposes
df['combined_text'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)  # This ensures that all relevant information from each row is included in the embedding process

# Convert each row of the DataFrame into a Document object
# Each Document contains the combined text as its content and metadata (e.g., row index) for reference
documents = [
    Document(page_content=' '.join(row.astype(str)), metadata={'index': i})
    for i, row in df.iterrows()
]
# Initialize a RecursiveCharacterTextSplitter to split long documents into smaller chunks
# Parameters:
#   - chunk_size: Maximum size of each chunk (in characters)
#   - chunk_overlap: Number of overlapping characters between consecutive chunks to maintain context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
# Split the documents into smaller chunks using the text splitter
texts = text_splitter.split_documents(documents)
# Initialize an embedding model from Google Generative AI
# This model will be used to generate embeddings for the text chunks
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a FAISS vector store to store the embeddings of the text chunks
# FAISS is a library for efficient similarity search and clustering of dense vectors
vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)

# Save the FAISS vector store locally for future use
# Replace the file path with the desired location to save the vector store
print("Saving the vector store locally...")
vectorstore.save_local("E:\\IMDB_MOVIES_PLATFORM\\faiss_index")


