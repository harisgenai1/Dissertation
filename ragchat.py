import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.messages import AIMessage, HumanMessage
import tempfile

# Load environment variables
load_dotenv()

# Initialize embedding and language model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm_instance = ChatOpenAI(model="gpt-3.5-turbo")

# Load and process documents
def load_and_process_docs(file_path):
    try:
        # Create a temporary directory to hold the file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "uploaded.pdf")

            # Copy the uploaded file to the temporary directory
            with open(temp_file_path, "wb") as temp_file:
                with open(file_path, "rb") as uploaded_file:
                    temp_file.write(uploaded_file.read())

            # Load and process the document from the temporary file
            loader = PyPDFDirectoryLoader(temp_dir)
            docs = loader.load()

            if not docs:
                st.error("No documents found in the uploaded file.")
                return []

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            if not chunks:
                st.error("No document chunks created.")
                return []

            return chunks

    except Exception as e:
        st.error(f"An error occurred while processing the document: {e}")
        return []

# Set up embedding and vector store
def vect_store(chucked_doc):
    if not chucked_doc:
        st.error("No document chunks found. Please check the uploaded file.")
        return

    try:
        vectorstore = FAISS.from_documents(chucked_doc, embedding)
        vectorstore.save_local("vectors_tore")
        st.success("Vector store created successfully!")
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")

# Set up prompt template
prompt_template = """
    You are the Expert Document summarizer chatbot. If the question is not related to the context, then act as an Assistant chatbot, ask and provide them topics which are in context and answer it.
    Provide Answer should include Topic Name, Answer the question pointwise based on the provided context only less than 300 words.
    
    <context>
    {context}
    </context>
    Question: {question}

    Assistant:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the chain for getting responses
def get_response(llm_instance, vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm_instance,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa_chain({"query": query})
    return result["result"]

def show_page():
    st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:")
    st.header("AI Chatbot")

    # File uploader for document upload
    uploaded_file = st.file_uploader("Upload your PDF or document", type=["pdf"])

    if uploaded_file:
        st.write("Document uploaded successfully. Creating vector store...")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.write(f"Temporary file path: {temp_file_path}")  # Debugging statement

        # Load and process the document
        chucked_doc = load_and_process_docs(temp_file_path)

        if chucked_doc:
            # Create vector store
            vect_store(chucked_doc)

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi, How can I assist you?"),
        ]

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Input for asking a question
    user_query = st.chat_input("Type your message here...")
    if user_query:
        # Append user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        # Generate and display response
        with st.spinner("Generating response..."):
            try:
                vectorstore = FAISS.load_local("vectors_tore", embedding, allow_dangerous_deserialization=True)
                response = get_response(llm_instance, vectorstore, user_query)
                st.session_state.chat_history.append(AIMessage(content=response))

                with st.chat_message("AI"):
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")

if __name__ == "__main__":
    show_page()
