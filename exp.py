from flask import Flask, request, render_template, jsonify
import base64
import os
from dotenv import load_dotenv
import validators
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

app = Flask(__name__)
load_dotenv()

# Function to encode an image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
img_file = r"C:\Users\shaik\Downloads\ai-chip-artificial-intelligence-future-technology-innovation.jpg"
side_file = r"C:\Users\shaik\OneDrive\Pictures\8-bit City_1920x1080.jpg"

# Encode the image
try:
    img_base64 = get_base64_of_bin_file(img_file)
    img_base642 = get_base64_of_bin_file(side_file)
except FileNotFoundError:
    img_base64 = None
    img_base642 = None

@app.route('/')
def home():
    return render_template('index.html', img_base64=img_base64, img_base642=img_base642)

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form.get('url')
    if not url.strip():
        return jsonify({"error": "Please enter a URL"}), 400
    elif not validators.url(url):
        return jsonify({"error": "Invalid URL. Please try again."}), 400
    else:
        try:
            if "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            else:
                loader = UnstructuredURLLoader(
                    urls=[url], ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}
                )
            data = loader.load()

            if data:
                llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                prompt_template = """
                Provide Title,Subject and Write a summary pointwise ,Anyhow complete the following speech less than 300 words of the following Text,
                speech: {text}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output = chain.run(data)
                return jsonify({"summary": output}), 200
            else:
                return jsonify({"error": "Failed to load content from the URL."}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            chunks = load_and_process_docs(temp_file_path)
            if chunks:
                vect_store(chunks)
                return jsonify({"message": "Vector store created successfully!"}), 200
            else:
                return jsonify({"error": "Failed to process the document."}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def load_and_process_docs(file_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "uploaded.pdf")
            with open(temp_file_path, "wb") as temp_file:
                with open(file_path, "rb") as uploaded_file:
                    temp_file.write(uploaded_file.read())

            loader = PyPDFDirectoryLoader(temp_dir)
            docs = loader.load()

            if not docs:
                return []

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            if not chunks:
                return []

            return chunks
    except Exception as e:
        return []

def vect_store(chunks):
    if not chunks:
        return

    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding)
        vectorstore.save_local("vectors_store")
    except Exception as e:
        return

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.form.get('query')
    if not user_query.strip():
        return jsonify({"error": "Please enter a query"}), 400

    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("vectors_store", embedding, allow_dangerous_deserialization=True)
        response = get_response(ChatOpenAI(model="gpt-3.5-turbo"), vectorstore, user_query)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_response(llm_instance, vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt_template = """
        You are the Expert Document summarizer chatbot. If the question is not related to the context, then act as an Assistant chatbot, 
        ask and provide them topics which are in context and answer it. Provide Answer should include Topic Name, Answer the question 
        pointwise based on the provided context only less than 300 words.
        
        <context>
        {context}
        </context>
        Question: {question}

        Assistant:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm_instance,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa_chain({"query": query})
    return result["result"]

@app.route('/longtext', methods=['POST'])
def longtext():
    doc = request.form.get('text')
    if not doc.strip():
        return jsonify({"error": "Please enter a text"}), 400

    try:
        document = Document(page_content=doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_docs = splitter.split_documents([document])

        if final_docs:
            llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-instruct")
            template = """Write a concise summary of the following:
                        {text}
                        CONCISE SUMMARY:"""
            prompt_template = PromptTemplate(template=template, input_variables=['text'])
            refine_template = """
            with the help of important points
            Provide Title,Subject and Write a Final summary with point wise  ,complete the following  entire speech less than 300 words  of the following Text,
            speech: {text}
            """
            refine_prompt = PromptTemplate(template=refine_template, input_variables=['text'])
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
            output = result["output_text"]
            return jsonify({"summary": output}), 200
        else:
            return jsonify({"error": "Failed to load content from the text."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
