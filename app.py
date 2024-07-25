import os
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

st.set_page_config(page_title="DocumentGPT", page_icon="📃")

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        cache_dir = LocalFileStore(f"./embeddings_cache/{file_name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"Error embedding file: {e}")
        return None

def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    if "messages" in st.session_state:
        for message in st.session_state["messages"]:
            send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            Context : {context}
            """
        ),
        ("human", "{question}")
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    Use this chatbot to ask questions to an AI about your files!
    """
)

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

if uploaded_file:
    with st.spinner("Processing file..."):
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        retriever = embed_file(file_path)
        if retriever:
            send_message("File uploaded and processed successfully! Ask away!", "ai", save=False)
        else:
            send_message("Failed to process the file. Please try again.", "ai", save=False)
else:
    send_message("Please upload a file to begin.", "ai", save=False)

paint_history()
message = st.chat_input("Ask anything about your file...")

if message and retriever:
    send_message(message, "human")
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    with st.chat_message("ai"):
        response = chain.invoke(message)
        


