# langchain 사용
# default: zero-short
# 여유가 된다면, 외부 api 사용
# 카톡에 연동시킬 것



import os
import json
import openai
import tkinter as tk
import pandas as pd
from collections import defaultdict
from tkinter import scrolledtext
import tkinter.filedialog as filedialog
import chromadb
openai.api_key = os.environ["openai_api_key"]

client = chromadb.PersistentClient()
collection = client.get_or_create_collection(
    name="data",
    metadata={"hnsw:space": "cosine"}
)

def read_data(filename = "project_data_카카오톡채널.txt"):
    filepath = os.path.join("./data", filename)
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines

def get_manual_dict(lines):
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

def get_reference_data(data_name="카카오톡채널"):
    data_name="카카오톡채널"
    filename = f"project_data_{data_name}.txt"
    lines = read_data(filename)
    manual_dict = get_manual_dict(lines)
    json_data = json.dumps(manual_dict)
    return manual_dict

def add_data_to_db(manual_data):
    ids = []
    doc_meta = []
    docs = []
    for key, val in manual_data.items():
        ids.append(key)
        docs.append(val)

    collection.add(
        documents=docs,
        ids=ids,
    )

def get_data_from_db(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
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

def db_initialize(data_name="카카오톡채널"):
    print(f"db initialize - {data_name}")
    filename = f"project_data_{data_name}.txt"
    lines = read_data(filename)
    manual_dict = get_manual_dict(lines)
    add_data_to_db(manual_dict)
    print("done")

def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
        # max_tokens=4096
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "get_data_from_db": get_data_from_db,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
            # max_tokens=4096
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def main():
    message_log = [
        {
            "role": "system",
            "content": f'''
            너는 고객의 질문에 대답을 하는 상담사야. 내용은 질문에 대한 대답은 function 을 통해서 찾도록 해.
            '''
        }
    ]

    functions = [
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
    
    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)

    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    db_initialize("카카오톡싱크")
    main()