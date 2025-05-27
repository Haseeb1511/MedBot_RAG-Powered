import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from src.vision_model import extract_text_from_image,create_vision_message

st.title("Medical Assistant ChatBot")

#--------------------------------------------------Vision Model------------------------------------------------------------------------------------
def vision_model():
    uploader = st.file_uploader("Upload your picture here ",type=["png", "jpg", "jpeg"])
    if uploader is not None:
        pic = Image.open(uploader)
        display_pic = st.sidebar.image(uploader,caption="Uploaded Image") 
        vision_query = st.text_input("Enter query")
        vision_model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

        image = extract_text_from_image(pic)
        message = create_vision_message(image,vision_query)
        if vision_query:
            vision_result = vision_model.invoke([message]).content
            st.markdown(vision_result)
            # Show download button
            st.download_button(label="Download Response as txt",data=vision_result,file_name="model respone.txt",mime="text/plain")

#-----------------------------------------------------------------Text_Based-Model--------------------------------------------------------------------------

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("enter query")
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user","content":query})


    response = "this is response"
    st.chat_message("ai").markdown(response)
    st.session_state.messages.append({"role":"user","content":response})








st.sidebar.info("Please chose the type of qwery from those given below:")
choice = st.sidebar.radio("Chose From Here:",["Vision","Dermatology","Psychiatrist","Dentist"])

if choice == "Vision":
    vision_model()
elif choice=="Dermatology":
    main()

elif choice == "Psychiatrist":
    pass

elif choice == "Dentist":
    pass










    # Show download button
    # st.download_button(
    #     label="Download as PNG",
    #     data=img_buffer,
    #     file_name="converted_image.png",
    #     mime="image/png"
    # )