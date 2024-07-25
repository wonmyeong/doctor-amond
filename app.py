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

st.set_page_config(
    page_title="PetGPT",
    page_icon="🐶",
)

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

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        cache_dir = LocalFileStore(f"/tmp/embeddings_cache/{file_name}")
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

st.title("PetGPT")

st.markdown(
    """
    안녕하세요 PetGPT입니다!!!!
    반려견에 관해 궁금한 점 있으시면 질문해 주세요🐕🐕🐕
    """
)

# 미리 업로드된 파일 경로 설정
file_path = "dog_health.txt"

# 파일이 존재하는지 확인
if os.path.exists(file_path):
    retriever = embed_file(file_path)
    if retriever:
        send_message("File loaded and processed successfully! Ask away!", "ai", save=False)
    else:
        send_message("Failed to process the file. Please try again.", "ai", save=False)
else:
    send_message("File not found. Please upload the file and try again.", "ai", save=False)

paint_history()
message = st.chat_input("메세지 PegGPT")

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


