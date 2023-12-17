import os

from database import ChromaDB
from langchain_openai import LangChain

class ChatBotCMD():
    def __init__(self, langchain):
        self.langchain = langchain
    
    def run(self):
        while True:
            user_input = input("user: ")
            self.langchain.add_message_log({"role": "user", "content": user_input})
            
            response = self.langchain.run()
            self.langchain.add_message_log({"role": "assistant", "content": response})
            
            print("AI: ", response)
    
if __name__ == "__main__":
    db = ChromaDB(data_name="카카오싱크")
    lc = LangChain(db=db)
    chatbot_cmd = ChatBotCMD(lc)
    chatbot_cmd.run()