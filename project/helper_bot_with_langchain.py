import os
import json
import openai
import chromadb
import pandas as pd
import tkinter as tk
from collections import defaultdict
from tkinter import scrolledtext

from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI


class DataBase():
    def __init__(self, data_name):
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(
            name="data",
            metadata={"hnsw:space": "cosine"}
        )
        self.data_dir = "./data"
        self.n_results = 3
        self.initialize(data_name)

    def read_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
        return lines

    def initialize(self, data_name):
        print(f"db initialize - {data_name}")
        filename = f"project_data_{data_name}.txt"
        lines = self.read_data(filename)
        manual_dict = self.get_manual_dict(lines)
        self.add_data_to_db(manual_dict)

    def get_manual_dict(self, lines):
        manual_dict = defaultdict(list)
        manual_key = ""
        for line in lines:
            line = line.replace("\n", "")
            if not line:
                continue

            if line[0] == "#":
                manual_key = line.replace("#", "")
            elif manual_key:
                manual_dict[manual_key].append(line)
            else:
                continue

        for key in manual_dict:
            manual_dict[key] = "\n".join(manual_dict[key])

        return manual_dict

    def add_data_to_db(self, manual_data):
        ids = []
        doc_meta = []
        docs = []
        for key, val in manual_data.items():
            ids.append(key)
            docs.append(val)

        self.collection.add(
            documents=docs,
            ids=ids,
        )

    def get_data_from_db(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=self.n_results,
        )
        print(f"db query result: {results}")

        ids = results["ids"][0]
        documents = results["documents"][0]

        ans_data = {}
        for i in range(len(ids)):
            ans_data[ids[i]] = documents[i]

        print("answer dict")
        for key, val in ans_data.items():
            print(f"{key}: {val}")

        return json.dumps(ans_data)


class LangChain():
    def __init__(self, api_key, db):
        self.db = db
        self.openai = openai
        self.openai.api_key = api_key
        self.llm = ChatOpenAI(temperature=0.8)
        self.chain = None

        self.system_message = "너는 고객의 질문에 대답을 하는 상담사야. 내용은 질문에 대한 대답은 function 을 통해서 찾도록 해."
        self.message_log = []
        self.functions = [
            {
                "name": "get_data_from_db",
                "description": "db에 저장된 데이터로부터 응답 도출",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]
    
    def add_message_log(self, msg):
        self.message_log.append(msg)

    def send_message(self, gpt_model="gpt-3.5-turbo", temperature=0.1):
        response = self.openai.ChatCompletion.create(
            model=gpt_model,
            messages=self.message_log,
            temperature=temperature,
            functions=self.functions,
            function_call='auto',
        )

        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            available_functions = {
                "get_data_from_db": self.db.get_data_from_db,
            }
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
            # **function_args로 처리하기
            function_response = fuction_to_call(**function_args)

            # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
            self.add_message_log(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
            self.add_message_log(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # 함수 실행 결과도 GPT messages에 추가하기
            response = self.openai.ChatCompletion.create(
                model=gpt_model,
                messages=self.message_log,
                temperature=temperature,
                # max_tokens=4096
            )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
        return response.choices[0].message.content

    def get_system_message_prompt(self):
        return SystemMessage(content=self.system_message)

    def get_human_message_prompt(self):
        return HumanMessagePromptTemplate.from_template("{text}")
    
    def set_chain(self):
        smp = self.get_system_message_prompt()
        hmp = self.get_human_message_prompt()
        chat_prompt = ChatPromptTemplate.from_messages([smp, hmp])
        self.chain = LLMChain(llm=self.llm, prompt=chat_prompt)
    
    def get_txt_from_message_log(self):
        txt = self.message_log[-1]["content"]
        return txt

    def run(self):
        self.set_chain()
        result = self.chain.run(text = self.get_txt_from_message_log())
        return result


class GUI():
    def __init__(self, langchain):
        self.langchain = langchain
        self.user_entry = None
        self.window = None
        self.conversation = None

    def run(self):
        self.window = tk.Tk()
        self.window.title("GPT AI")

        font = ("맑은 고딕", 10)

        self.conversation = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, bg='#f0f0f0', font=font)
        # width, height를 없애고 배경색 지정하기(2)
        self.conversation.tag_configure("user", background="#c9daf8")
        # 태그별로 다르게 배경색 지정하기(3)
        self.conversation.tag_configure("assistant", background="#e4e4e4")
        # 태그별로 다르게 배경색 지정하기(3)
        self.conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # 창의 폭에 맞추어 크기 조정하기(4)

        input_frame = tk.Frame(self.window)  # user_entry와 send_button을 담는 frame(5)
        input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

        self.user_entry = tk.Entry(input_frame)
        self.user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

        send_button = tk.Button(input_frame, text="Send", command=self.on_send)
        send_button.pack(side=tk.RIGHT)

        self.window.bind('<Return>', lambda event: self.on_send())
        self.window.mainloop()

        
    def show_popup_message(self, message):
        popup = tk.Toplevel(self.window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        self.window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = self.window.winfo_x()
        window_y = self.window.winfo_y()
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(self.window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send(self):
        user_input = self.user_entry.get()
        self.user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            self.window.destroy()
            return

        self.langchain.add_message_log({"role": "user", "content": user_input})
        self.conversation.config(state=tk.NORMAL)  # 이동
        self.conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = self.show_popup_message("처리중...")
        self.window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = self.langchain.run()
        thinking_popup.destroy()

        self.langchain.add_message_log({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        self.conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        self.conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        self.conversation.see(tk.END)

        

if __name__ == "__main__":
    db = DataBase(data_name="카카오싱크")
    lc = LangChain(api_key=os.environ["openai_api_key"], db=db)
    gui = GUI(lc)
    gui.run()