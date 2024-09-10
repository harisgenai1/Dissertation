import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
# Custom CSS to change the background color


def show_page():
    load_dotenv()
    os.environ.get("OPENAI_API_KEY")

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    prompt_template = """
    Provide Title,Subject and Write a  summary pointwise ,Anyhow complete the following speech less than 300 words  of the following Text,
    speech: {text}
   
    """
    

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    st.title("Youtube Website Summarizer :robot_face:")
    st.subheader("Summarize URL")

    url = st.text_input("Enter the URL")

    if st.button("Summarize"):
        if not url.strip():
            st.error("Please enter a URL")
        elif not validators.url(url):
            st.error("Invalid URL. Please try again.")
        else:
            try:
                with st.spinner("Loading content..."):
                    if "youtube.com" in url:
                        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[url], ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}
                        )
                    data = loader.load()

                    if data:
                        chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                        output = chain.run(data)
                        st.write(output)
                    else:
                        st.error("Failed to load content from the URL.")
            except Exception as e:
                st.exception(f"Exception: {e}")

                
