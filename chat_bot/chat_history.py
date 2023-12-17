import os
import json
import openai
from langchain.memory import FileChatMessageHistory

class ChatHistory():
    def __init__(self, max_history=10, history_dir = "./chat_histories", conversation_id = "test"):
        self.conversation_id = conversation_id
        self.history_dir = history_dir
        self.history = None
        self.conversation = []
        self.role_dict = {"Human": "user", "ai": "assistant"}
        self.max_history = max_history
        self.initialize()

    def initialize(self):
        self.init_history()
        self.init_conversation()
        
    def init_history(self):
        os.makedirs(self.history_dir, exist_ok=True)
        history_filepath = os.path.join(self.history_dir, f"{self.conversation_id}.json")
        self.history = FileChatMessageHistory(history_filepath)

    def init_conversation(self):
        filepath = os.path.join(self.history_dir, self.conversation_id)
        if not os.path.exists(filepath):
            conversation_list = []
        else:
            with open(filepath, "r") as f:
                conversation_list = json.load(f)

        for conv in conversation_list:
            self.add_conversation(conv["type"], conv["data"]["content"])
    
    def add_history(self, role, content):
        if role=="Human":
            self.history.add_user_message(content)
        if role=="ai":
            self.history.add_ai_message(content)

    def add_conversation(self, role, content):
        self.conversation.append(
                {
                    "role": self.role_dict[role],
                    "content": content
                }
            )
        self.conversation = self.conversation[-self.max_history:]

    def update_history(self, role="Human", content=""):
        self.add_history(role, content)
        self.add_conversation(role, content)

    def send_message(self, msg, gpt_model="gpt-3.5-turbo", temperature=0.1):
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=msg,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def generate_answer_with_history(self, user_message):
        self.update_history(role="Human", content=user_message)
        answer = self.send_message(self.conversation)
        self.update_history(role="ai", content=answer)

        return answer
    
    def run(self):
        while True:
            msg = input("user: ")
            answer = self.generate_answer_with_history(msg)
            print("AI: ", answer)

if __name__ == "__main__":
    test = ChatHistory(conversation_id="test")
    test.run()