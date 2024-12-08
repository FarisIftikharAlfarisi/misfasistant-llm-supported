#import streamlit
import streamlit as st

#langchain import
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

#groq
from groq import Groq

#load env
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
#config page
st.set_page_config(
    page_title="Misfasistant",
    page_icon="kaaba:",
    layout="centered",
)

#app section 
st.title(':kaaba: Misfasistant')
st.write("Adaptasi teknologi untuk membantu anda dalam perjalanan religi :smile:")


GROQ_API_KEY = os.getenv("API_KEY")
CHAT_MODEL = os.getenv("MODEL")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

def load_and_process_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        ).from_loaders([loader])
        return index
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        text_content = df.to_string(index=False)
        return text_content
    else:
        raise ValueError("Unsupported file type. Please use PDF or Excel.")

def generate_response(prompt, document_content):
    if isinstance(document_content, str):  # If Excel content (string)
        context = f"File Content:\n{document_content}\n\nQuestion: {prompt}\nAnswer:"
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant named "misfasistant". Respond in Indonesian. respond politely indonesian. you will help user to know our about umrah package. you will help user to know our about umrah company.you will help user to find best umrah package suits with their budget. if theres no umra package fit with user budget, just tell user to contact salesperson that we have in data. Develop the data in the Excel file into a narrative format to make it easier for users to read. if you found true or false on columns it means users wont have that feature on their umrah package in other words that feature not included on the package.If the user asks about a specific Umrah package,just provide the salesperson data according to the corresponding row.'},
                {'role': 'user', 'content': context}
            ]
        )
        return response.choices[0].message.content
    else:  
        qa_chain = RetrievalQA.from_chain_type(
            llm=client,
            retriever=document_content.as_retriever(),
            chain_type="stuff"
        )
        return qa_chain.run(prompt)

file_path = "data.xlsx"  
document_content = load_and_process_document(file_path)

#setup session
if 'user_messages' not in st.session_state:
    st.session_state.user_messages = []

for message in st.session_state.user_messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('chat here...')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.user_messages.append({'role': 'user', 'content': prompt})
    
    assistant_response = generate_response(prompt, document_content)
    st.session_state.user_messages.append({'role': 'assistant', 'content': assistant_response})
  
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
