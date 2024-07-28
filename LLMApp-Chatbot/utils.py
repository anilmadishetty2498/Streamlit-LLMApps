# Databricks notebook source
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
import pinecone
from langchain.vectorstores import Chroma


#Function to fetch data from website
#https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/sitemap
def get_website_data(sitemap_url):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(sitemap_url)
    docs = loader.load()

    return docs

#Function to split data into smaller chunks
def split_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

#Function to create embeddings instance
def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


#Function to push data to Pinecone
def push_to_pinecone(pinecone_index_name,embeddings, docs):
    index = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return index

#Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_index_name,embeddings):
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    return index

#This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs