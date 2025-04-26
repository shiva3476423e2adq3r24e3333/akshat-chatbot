import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

import os

st.set_page_config(page_title="Akshat Chatbot")

st.title("🤖 Ask Akshat Shrivastava’s Community Posts")
uploaded_file = st.file_uploader("Upload `akshat_posts.txt` file", type="txt")

if uploaded_file:
    text = uploaded_file.read().decode()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    st.success("✅ Posts indexed! Ask anything below 👇")

    query = st.text_input("What do you want to know?", placeholder="e.g. What was Akshat's view on PVR?")
    
    if query:
        llm = ChatOpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = vectorstore.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.markdown(f"**Answer:** {response}")
