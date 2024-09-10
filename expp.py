import streamlit as st
import base64

# Function to encode an image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
img_file = r"C:\Users\shaik\OneDrive\Pictures\8-bit City_1920x1080.jpg"

# Encode the image
img_base64 = get_base64_of_bin_file(img_file)

# Custom CSS to add a background image
background_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

# Inject the custom CSS
st.markdown(background_image, unsafe_allow_html=True)

st.title("Welcome to My Streamlit App")
st.write("This is an example of a custom background image using a local file.")
