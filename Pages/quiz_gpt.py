import json
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.schema import BaseOutputParser, output_parser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            st.error(f"JSONDecodeError: {e}")
            st.error(f"텍스트: {text}")
            raise

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
    Translate four questions into Korean.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "1 바다의 색깔은 무슨 색입니까?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "2 조지아의 수도 이름은 무엇입니까?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "3 영화 아바타는 언제 개봉했습니까?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "4 Julius Caesar가 누구입니까?",
                "answers": [
                        {{
                            "answer": "로마 황제",
                            "correct": true
                        }},
                        {{
                            "answer": "화가",
                            "correct": false
                        }},
                        {{
                            "answer": "배우",
                            "correct": false
                        }},
                        {{
                            "answer": "모델",
                            "correct": false
                        }}
                ]
            }}
           
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

def split_file(file_content, file_name):
    cache_dir = "./.cache/quiz_files/"
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="퀴즈 생성 중...", persist=True)
def run_quiz_chain(_docs,key):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

# 퀴즈 주제를 선택하는 selectbox
quiz_mode = st.selectbox(
    "퀴즈 주제를 선택해주세요",
    ["반려견 건강", "반려견 행동", "기타 상식"]
)

# 선택한 주제에 따라 파일 경로 설정
if quiz_mode == "반려견 건강":
    file_path = ".cache/files/health_quiz.txt"
elif quiz_mode == "반려견 행동":
    file_path = ".cache/files/behavior_quiz.txt"
else:
    file_path = ".cache/files/knowledge_quiz.txt"

# 파일 내용을 읽어옴
with open(file_path, "rb") as file:
    file_content = file.read()

# 파일 내용을 처리 함수에 전달
docs = split_file(file_content, os.path.basename(file_path))

# 퀴즈 생성 실행
response = run_quiz_chain(docs, key=quiz_mode)

# 질문과 답변을 표시하고 채점하는 폼 생성
with st.form("questions_form"):
    for idx, question in enumerate(response["questions"]):
        st.write(question["question"])
        value = st.radio(
            f"선택해주세요 {idx+1}",
            [answer["answer"] for answer in question["answers"]],
            key=f"{idx}_radio",
            index=None
        )
        if {"answer": value, "correct": True} in question["answers"]:
            st.success("정답입니다!")
        elif value is not None:
            st.error("틀렸습니다.")
    st.form_submit_button("제출")
