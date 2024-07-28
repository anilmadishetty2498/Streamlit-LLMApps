# Databricks notebook source
from langchain.vectorstores import Pinecone
#from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
#from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub

# COMMAND ----------

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# COMMAND ----------

def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
            #metadata={"name": filename.name,"id":filename.id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs

# COMMAND ----------

#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# COMMAND ----------

#Function to push data to Pinecone
def push_to_pinecone(pinecone_index_name,embeddings, docs):
    index = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return index

#Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_index_name,embeddings):
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    return index

# COMMAND ----------

def similar_docs(query,k,pinecone_index_name,embeddings,unique_id):
    index = pull_from_pinecone(pinecone_index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


# COMMAND ----------

def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary

# COMMAND ----------

def get_summary(current_doc):
    #llm = OpenAI(temperature=0)
    llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary

# COMMAND ----------

# This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k,  unique_id):
    similar_docs = index.similarity_search(query, int(k), {"unique_id": unique_id})
    return similar_docs