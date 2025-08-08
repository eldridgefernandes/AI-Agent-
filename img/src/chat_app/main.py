#%%
import argparse
import os
import shutil
import torch
import pandas as pd
import re

from langchain.schema.document import Document
from langchain.document_loaders import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dataclasses import dataclass
from typing import List


#%% Cell to load csv database
loader = CSVLoader(file_path="img\src\Data\dataset.csv")
documents = loader.load()
print(documents[0].page_content)

df = pd.read_csv("img\src\Data\dataset.csv")
df.to_csv('output.txt', sep='\t', index=False, header=False)
print(df)

# Function to convert PDF to text and append to vault.txt
def convert_csv_to_text():
    if True:
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', df).strip()
         # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
        chunks = []
        current_chunk = ""
        for sentence in sentences:
        # Check if the current sentence plus the current chunk exceeds the limit
            if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                # When the chunk exceeds 1000 characters, store it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence + " "
                if current_chunk:  # Don't forget the last chunk!
                    chunks.append(current_chunk)

            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
# Create a Chroma vector store
doc = "output.txt"
vectorstore = Chroma.from_texts(doc, embeddings, persist_directory="./chroma_db")
# Persist the vector store
vectorstore.persist()

query = "Your search query here"
results = vectorstore.similarity_search(query)

# Process and display results
for doc in results:
    print(doc.page_content)

#%%
# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='mistral'
)

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    print ("inside relevant block")
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
        # Encode the user input
    input_embedding = model.encode([user_input])
        # Compute cosine similarity between the input and vault embeddings
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    print(relevant_context)
    return relevant_context

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    if relevant_context:
    # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + context_str)
    else:
        print("No relevant context found.")
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input

    # Create a message history including the system message and the user's input with context
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input_with_context}
    ]
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model="mistral",
        messages=messages
    )
        # Return the content of the response from the model
    return response.choices[0].message.content

model = SentenceTransformer("all-MiniLM-L6-v2")
vault_content = []
if os.path.exists("output.txt"):
    with open("output.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

vault_embeddings = model.encode(vault_content) if vault_content else []

# Convert to tensor and print embeddings
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)
    
user_input = input("Ask a question about your documents: ")
system_message = "You are a helpful assistat that is an expert at extracting the most useful information from a given text"
response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model)
print("Mistral Response: \n\n" + response )