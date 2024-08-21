import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (
    load_gemini_flash_model,  # Updated to load the new model
    gemini_flash_response,  # Updated function name
    gemini_flash_vision_response,  # Updated function name
    embeddings_model_response
)

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Gemini AI',
                           ['ChatBot',
                            'Image Captioning',
                            'Embed text',
                            'Ask me anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                           default_index=0
                           )


# Function to translate roles between Gemini-Flash and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


# Chatbot page
if selected == 'ChatBot':
    model = load_gemini_flash_model()  # Updated to use the new model

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:  # Renamed for clarity
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini...")  # Updated prompt
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Flash and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)  # Renamed for clarity

        # Display Gemini-Flash's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)


# Image captioning page
if selected == "Image Captioning":

    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption"):
        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            default_prompt = "write a short caption for this image"  # Change this prompt as per your requirement

            # Get the caption of the image from the gemini-1.5-flash model
            caption = gemini_flash_vision_response(default_prompt, image)

            with col2:
                st.info(caption)


# Text embedding model
if selected == "Embed text":

    st.title("🔡 Embed Text")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        response = embeddings_model_response(user_prompt)
        st.markdown(response)


# Ask me anything page
if selected == "Ask me anything":

    st.title("❓ Ask me a question")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = gemini_flash_response(user_prompt)  # Updated function name
        st.markdown(response)