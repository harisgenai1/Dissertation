
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain 
from langchain_openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.documents import Document
import streamlit as st

def show_page():
    load_dotenv()
    os.environ.get("OPENAI_API_KEY")
    template = """Write a concise summary of the following:
                {text}
                CONCISE SUMMARY:"""

    prompt_template = PromptTemplate(template=template,input_variables =['text'])

    refine_template = (
    """with the help of important points
    Provide Title,Subject and Write a Final summary with point wise  ,complete the following  entire speech less than 300 words  of the following Text,
    speech: {text}
   
    """
    
   
    )
    refine_prompt = PromptTemplate(template=refine_template,input_variables = ['text'])


    llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-instruct")

    st.title("Text Summarizer :robot_face:")
    st.subheader(" Long Text Summarize ")

    doc = st.text_input("Enter the Text")

    if st.button("Summarize"):
        if not doc.strip():
            st.error("Please enter a text")
        
        else:
            try:
                with st.spinner("Loading content..."):
                    document = Document(
                                        page_content= doc
                                        )
                    # Create a RecursiveCharacterTextSplitter instance
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


                    final_docs = splitter.split_documents([document])

                    if final_docs:
                        chain = load_summarize_chain(
                                llm=llm,
                                chain_type="refine",
                                question_prompt=prompt_template,
                                refine_prompt=refine_prompt,
                                return_intermediate_steps=True,
                                input_key="input_documents",
                                output_key="output_text",
                                )
                        result = chain({"input_documents": final_docs}, return_only_outputs=True)

                        output =result["output_text"]
                        st.success(output)
                    else:
                        st.error("Failed to load content from the text.")
            except Exception as e:
                st.exception(f"Exception: {e}")
