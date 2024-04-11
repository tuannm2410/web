import streamlit as st
from PIL import Image
import streamlit as st
from img2text import create_description
from chatbot import create_text
import text2img 

def load_image(image_file):
	img = Image.open(image_file)
	return img

def convert_img_to_text():
    st.title("Image to Text Converter")
    image_file1 = st.file_uploader("Upload Images", type=["png","jpg","jpeg"],key=f'{1}')
    if image_file1 is not None:
            text = create_description(image_file1)
            # To See details
            st.subheader("Description")
            st.write(text)
            # To View Uploaded Image
            st.image(load_image(image_file1),width=500)

def uploader_callback():
    global question
    st.session_state.widget = ""
    question = st.session_state.get('widget', '')

def visual_question_answering():
    
    global question
    st.title("Visual Question Answering")
    image_file2 = st.file_uploader("Upload Images", type=["png","jpg","jpeg"],key=f'{2}',on_change=uploader_callback)

    if image_file2 is not None:
        cap_img = create_description(image_file2)
        st.text_input("Enter question for the picture",key= 'widget')
        question = st.session_state.get('widget', '')
        st.image(load_image(image_file2),width=500)
        if len(question) != 0:
            prompt = question + " answer this question for a picture with the caption: " + cap_img
            text_output = create_text(prompt)
            st.subheader("Answer ")
            st.write(text_output)


def image_generator_from_text():
    st.title("Text To Image ")    
    st.text_input("Enter description for the picture",key= 'widget1')
    prompt = st.session_state.get('widget1', '')   
    if len(prompt) != 0:
        img = text2img.create_img(prompt)
        st.image(img,width=500)



def image_generator_from_image():
    global cap_img
    st.title("Image To Image") 
    image_file3 = st.file_uploader("Upload Images", type=["png","jpg","jpeg"],key=f'{3}')
    if image_file3 is not None:
        cap_img = create_description(image_file3)
        st.image(load_image(image_file3),width=500)
        img = text2img.create_img(cap_img)
        st.subheader("The image created")
        st.image(img,width=500)

######
######
menu = ["Text To Image ","Image Captioning","Visual Question Answering","Image To Image"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Image Captioning":
    convert_img_to_text()

if choice == "Image To Image":
    image_generator_from_image()

if choice == "Text To Image ":
    image_generator_from_text()

elif choice == "Visual Question Answering":
    visual_question_answering()


