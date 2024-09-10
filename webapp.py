# app.py

import streamlit as st
import base64
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Sidebar navigation imports
import text_summarizer  # Ensure you have this module
import longtext_summarization  # Ensure you have this module
import chatbot  # Import the RAG Chatbot module

# Function to encode an image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
# Ensure this path is correct or use an image in your project directory
img_file = r"C:\Users\shaik\Downloads\hhhh.jpg"
side_file = r"C:\Users\shaik\OneDrive\Pictures\8-bit City_1920x1080.jpg"

# Encode the image
try:
    img_base64 = get_base64_of_bin_file(img_file)
    img_base642 = get_base64_of_bin_file(side_file)
except FileNotFoundError:
    st.error(f"Background image not found at {img_file}. Please check the path.")






# Custom CSS to add a background image and other styling
if 'img_base64' in locals():
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
           
        }}
       

        /* Change the font family for the entire app */
        body {{
            font-family: 'Roboto', sans-serif;
            color: #333333;
        }}
      


        /* Customize font for headers */
        h2,h3, h4, h5, h6  {{
            font-family: 'Niconne', cursive;
            text-align: center;
            font-size: 40px;
            font-weight: 600;
            color: #fdfdfe;
            text-shadow: 0px 0px 15px #b393d3, 0px 0px 20px #b393d3, 0px 0px 20px #b393d3,
                0px 0px 30px #b393d3;
            }}
        
        
        p {{
            font-family: 'Niconne', cursive;
            
            font-size: 20px;
            font-weight: 600;
            color: #fdfdfe;
            text-shadow: 0px 0px 15px #b393d3, 0px 0px 20px #b393d3, 0px 0px 20px #b393d3,
                0px 0px 30px #b393d3;
            }}
        li {{
            font-family: 'Niconne', cursive;
            
            font-size: 20px;
            font-weight: 600;
            color: #fdfdfe;
            text-shadow: 0px 0px 15px #b393d3, 0px 0px 20px #b393d3, 0px 0px 20px #b393d3,
                0px 0px 30px #b393d3;
            }}
        #bui1 {{
                background: #000000;
                border: 5px solid white;
            }}
        
        .st-emotion-cache-1v6glgu ,.st-emotion-cache-1v6glgu > ul[role="listbox"]:not(:last-child),e16jpq801, .st-emotion-cache-uct3qm,.st-emotion-cache-zvs45k{{
        background: #000000;
        }}

        .st-emotion-cache-arzcut,.st-emotion-cache-uhkwx6 {{
            
            background-color: #6225E6;
                }}       
        span {{
            font-family: 'Niconne', cursive;
            
            font-size: 20px;
            font-weight: 600;
            color: #fdfdfe;
            text-shadow: 0px 0px 15px #b393d3, 0px 0px 20px #b393d3, 0px 0px 20px #b393d3,
                0px 0px 30px #b393d3;
            }}
        .st-emotion-cache-1c7y2kd{{
            text-align: end ;
            background-color: rgb(108 26 237 / 50%);
        }}
        .st-emotion-cache-4oy321{{
            background-color: rgb(37 201 11 / 50%);
        }}
        .st-emotion-cache-6qob1r{{
            background: url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
            

        }}
        .st-aq {{
            margin-top: 100px;

            }}
        .st-aq:hover{{
            background-color: #b393d3;
            border: 5px solid #ffffff;
            border-radius: 10px;
        }}
        .st-emotion-cache-12fmjuu {{
            
            background: #6225E6;
            }}
        .st-e1{{
            font-family: 'Niconne', cursive;
            
            font-size: 20px;
            font-weight: 600;
            color: #fdfdfe;
            text-shadow: 0px 0px 15px #b393d3, 0px 0px 20px #b393d3, 0px 0px 20px #b393d3,
                0px 0px 30px #b393d3;
            }}
        
        .css-1v3fvcr {{
            font-family: 'Niconne', cursive;
            
            font-size: 20px;
            font-weight: 600;
            color: #fdfdfe;
            text-shadow: 0px 0px 15px #b393d3, 0px 0px 20px #b393d3, 0px 0px 20px #b393d3,
                0px 0px 30px #b393d3;
            }}

        
        
        

        
        
        h1{{
            font-size: 70px;
            text-align: center;
            font-weight: 600;
            font-family: 'Roboto', sans-serif;
            color: #b393d3;
            text-transform: uppercase;
            text-shadow: 1px 1px 0px #6225E6,
                        1px 2px 0px #6225E6,
                        1px 3px 0px #6225E6,
                        1px 4px 0px #6225E6,
                        1px 5px 0px #6225E6,
                        1px 6px 0px #6225E6,
                        1px 10px 5px rgba(16, 16, 16, 0.5),
                        1px 15px 10px rgba(16, 16, 16, 0.4),
                        1px 20px 30px rgba(16, 16, 16, 0.3),
                        1px 25px 50px rgba(16, 16, 16, 0.2);
            }}
        
        input[type="text"] {{
            font-family: 'Verdana', sans-serif !important;
            font-size: 22px !important;
            color: #000000;
        }}

    

      


       
        </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["YouTube Summarizer", "RAG Chatbot", "Long Text Summarizer"])

# Navigate to the selected page
if page == "YouTube Summarizer":
    text_summarizer.show_page()  # Ensure 'text_summarizer.py' has a 'show_page()' function
elif page == "RAG Chatbot":
    chatbot.show_page()  # Call the 'show_page()' function from 'chatbot.py'
elif page == "Long Text Summarizer":
    longtext_summarization.show_page()  # Ensure 'longtext_summarization.py' has a 'show_page()' function




# Custom CSS to style the button
import streamlit as st

# Custom CSS to style the button with animation
st.markdown("""
    <style>
    * {
      box-sizing: border-box;
    }

    /* Additional styling for buttons, inputs, etc. */
    .stButton>button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 10px 45px;
        font-family: 'Courier New', monospace;
        font-size: 18px;
        color: white;
        background: #6225E6;
        transition: 0.5s;
        box-shadow: 6px 6px 0 black;
        transform: skewX(-15deg);
        border: none;
        cursor: pointer;
    }
    

    .stButton>button:focus {
        outline: none; 
    }

    .stButton>button:hover {
        box-shadow: 10px 10px 0 #FBC638;
    }

    .stButton>button span {
        transform: skewX(15deg);
    }

    .stButton>button span:nth-child(2) {
        display: inline-block;
        width: 20px;
        margin-left: 30px;
        position: relative;
        top: 2px;
        transition: 0.5s;
        transform: translateX(-30%);
    }

    .stButton>button:hover span:nth-child(2) {
        margin-right: 45px;
        transform: translateX(0%);
    }

    /**************SVG****************/
    .stButton>button svg {
        width: 20px;
        height: 20px;
        margin-left: 10px;
    }

    .stButton>button path.one {
        transition: 0.4s;
        transform: translateX(-60%);
    }

    .stButton>button path.two {
        transition: 0.5s;
        transform: translateX(-30%);
    }

    .stButton>button:hover path.one {
        transform: translateX(0%);
        animation: color_anim 1s infinite 0.6s;
    }

    .stButton>button:hover path.two {
        transform: translateX(0%);
        animation: color_anim 1s infinite 0.4s;
    }

    .stButton>button:hover path.three {
        animation: color_anim 1s infinite 0.2s;
    }

    /* SVG animations */
    @keyframes color_anim {
        0% {
            fill: white;
        }
        50% {
            fill: #FBC638;
        }
        100% {
            fill: white;
        }
    }
    </style>
""", unsafe_allow_html=True)



