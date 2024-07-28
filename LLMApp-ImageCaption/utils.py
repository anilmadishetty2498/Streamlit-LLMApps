# Databricks notebook source
from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor
import torch

# COMMAND ----------

model_name = "Salesforce/blip-image-captioning-base"

model_bf16 = BlipForConditionalGeneration.from_pretrained(
                                               model_name,
                               torch_dtype=torch.bfloat16
)

# COMMAND ----------

processor = BlipProcessor.from_pretrained(model_name)

# COMMAND ----------

results_bf16 = get_generation(model_bf16, 
                              processor, 
                              image, 
                              torch.bfloat16)

# COMMAND ----------

print("Image Captioning :\n", results_bf16)

# COMMAND ----------

def image_captioning():
    model_name = "Salesforce/blip-image-captioning-base"

    model_bf16 = BlipForConditionalGeneration.from_pretrained(
                                                model_name,
                                torch_dtype=torch.bfloat16)
    processor = BlipProcessor.from_pretrained(model_name)

    results_bf16 = get_generation(model_bf16, 
                              processor, 
                              image, 
                              torch.bfloat16)
    
    return results_bf16
    