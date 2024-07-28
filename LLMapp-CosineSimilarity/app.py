# Databricks notebook source
import os
import streamlit as st
from langchain.schema import Document 
from langchain_community.vectorstores import FAISS #. # Chroma, Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Educate", page_icon=":books:")
#st.title("Vector Search")
st.subheader('Ask me something, will give out similar things')

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

#load csv file 
# from langchain.document_loaders.csv_loader import CSVLoader
# loader = CSVLoader(file_path='myData.csv', csv_args={'delimiter':',',
#                                                      'quotechar':'"',
#                                                      'fieldnames':['Words']})

# data = loader.load()

#db = FAISS.from_documents(documents=data,  embedding=embeddings)

# OR 

def load_data(file_path):
    return pd.read_csv(file_path, delimiter=',', quotechar='"')

data = load_data('myData.csv')

documents = [Document(page_content=text) for text in data['Words'].tolist()]

db = FAISS.from_documents(documents=documents,  embedding=embeddings)

def get_text():
    input_text = st.text_input("You will be asked a question: ", value="input")
    return input_text

user_input = get_text()

submit = st.button('Find similar things')

if submit:
    docs = db.similarity_search(user_input)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)  
    #st.write(docs)