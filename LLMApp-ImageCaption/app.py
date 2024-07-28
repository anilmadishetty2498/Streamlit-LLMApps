# Databricks notebook source
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
#from utils import *

from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor
from helper import load_image, get_generation
import torch

# COMMAND ----------

def main():
    load_dotenv()
    pinecone_index_name='resume'

    st.set_page_config(page_title="Images Captioning")
    st.title("Image Captioning ...💁")
    #st.subheader("I can help with image processing...💁")

    # Title of the Streamlit app
    st.subheader("Upload and Display Images")

    # Instructions for users
    st.write("Please upload an image or multiple images:")

    # job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",key="1")
    # document_count = st.text_input("No.of 'RESUMES' to return",key="2")
    # Upload the Resumes (pdf files)
    uploaded_files = st.file_uploader("Choose image/s...", type=["jpg", "jpeg", "png"],accept_multiple_files=True)

    model_name = "Salesforce/blip-image-captioning-base"

    model_bf16 = BlipForConditionalGeneration.from_pretrained(
                                                model_name,
                                    torch_dtype=torch.bfloat16)
    
    processor = BlipProcessor.from_pretrained(model_name)

    #https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg

    submit=st.button("Submit")

    if submit:
        with st.spinner('Wait for it...'):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Open the image file
                    image = Image.open(uploaded_file)
                    st.image(image)


                    results_bf16 = get_generation(model_bf16, 
                                                  processor, 
                                                  image, 
                                                  torch.bfloat16)
                    
                    #st.write(results_bf16)
                    st.subheader("👉 "+str(results_bf16))
                    
                st.success("Hope the image captioning was appropriate ❤️")

            else:
                st.markdown("<p style='font-weight: bold; color: red;'>No files uploaded yet. Please upload an image.</p>", unsafe_allow_html=True)

    #st.success("Hope I was able to save your time❤️")

# COMMAND ----------

# Run the Streamlit app
if __name__ == "__main__":
    main()