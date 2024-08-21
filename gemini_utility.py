import os
import json
from PIL import Image
import google.generativeai as genai

# Working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path of config_data file
config_file_path = f"{working_dir}/config.json"
config_data = json.load(open("config.json"))

# Loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini 1.5 Flash model
def load_gemini_flash_model():
    gemini_flash_model = genai.GenerativeModel("gemini-1.5-flash")
    return gemini_flash_model

# Get response from the updated model (gemini-1.5-flash) - image/text to text
def gemini_flash_vision_response(prompt, image):
    gemini_flash_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_flash_model.generate_content([prompt, image])
    result = response.text
    return result

# Get response from embeddings model - text to embeddings
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list

# Get response from Gemini 1.5 Flash model - text to text
def gemini_flash_response(user_prompt):
    gemini_flash_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_flash_model.generate_content(user_prompt)
    result = response.text
    return result

# Example usage:
# result = gemini_flash_response("What is Machine Learning")
# print(result)
# print("-"*50)
#
# image = Image.open("test_image.png")
# result = gemini_flash_vision_response("Write a short caption for this image", image)
# print(result)
# print("-"*50)
#
# result = embeddings_model_response("Machine Learning is a subset of Artificial Intelligence")
# print(result)