import os
import json
import openai

from datetime import datetime

from database import VectorDB, ChromaDB
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI

from chat_history import ChatHistory

prompt_dir = "./prompt_template"
prompt_template = {
    "intent_list": os.path.join(prompt_dir, "intent_list.txt"),
    "parse_intent": os.path.join(prompt_dir, "parse_intent.txt"),
    "default_response": os.path.join(prompt_dir, "default_response.txt"),
    "question_response": os.path.join(prompt_dir, "question_response.txt"),
    "qyery_result_check": os.path.join(prompt_dir, "query_result_check.txt"),
    "query_result_compression": os.path.join(prompt_dir, "query_result_compression.txt"),
    "function_use_check": os.path.join(prompt_dir, "function_use_check.txt"),
}
def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

class LangChain():
    def __init__(self, db=None, api_key = os.environ["OPENAI_API_KEY"]):
        self.db = db
        self.openai = openai
        self.openai.api_key = api_key
        self.llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        self.chain = None

        self.initialize()
    
    def initialize(self):
        self.set_chains()
        self.set_chat_history()
    
    def set_chains(self):
        self.intent_chain = self.create_chain_from_template(prompt_template["parse_intent"], "intent")
        self.default_chain = self.create_chain_from_template(prompt_template["default_response"], "output")
        self.question_chain = self.create_chain_from_template(prompt_template["question_response"], "output")
        self.query_result_check_chain = self.create_chain_from_template(prompt_template["qyery_result_check"], "output")
        self.query_result_compression_chain = self.create_chain_from_template(prompt_template["query_result_compression"], "output")
        self.function_use_check_chain = self.create_chain_from_template(prompt_template["function_use_check"], "output")

        self.intent_list = read_prompt_template(prompt_template["intent_list"])

    def set_chat_history(self):
        now = datetime.now()
        date_string = now.strftime("%Y%m%d_%H%M%S")
        self.chat_history = ChatHistory(conversation_id=date_string)

    def create_chain_from_template(self, template_path, output_key):
        chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_template(
                template=read_prompt_template(template_path)
            ),
            output_key=output_key,
            verbose=True,
        )
        return chain
    
    def get_data_from_db(self, user_message):
        context = {"user_message": user_message}
        context["query_results"] = self.db.query_db(user_message)
        has_value = self.query_result_check_chain.run(context)
        print("value: ", has_value)
        if has_value == "Y":
            return self.query_result_compression_chain.run(context)
        else:
            return ""

    def generate_answer(self, user_message):
        context = dict(user_message=user_message)
        context["chat_history"] = self.chat_history.conversation
        context["user_message"] = user_message

        context["intent_list"] = self.intent_list
        intent = self.intent_chain.run(context)

        if intent == "greeting":
            answer = self.default_chain.run(context)
        else:
            function_name = self.function_use_check_chain.run(context)
            print("function name: ", function_name)
            if function_name == "get_data_from_db":
                related_documents = self.get_data_from_db(user_message)
            else:
                related_documents = ""

            context["related_documents"] = related_documents
            print("related documents")
            print(context["related_documents"])
            answer = self.question_chain.run(context)

        self.chat_history.update_history(role="Human", content=user_message)
        self.chat_history.update_history(role="ai", content=answer)

        return answer

    def TEST(self):
        while True:
            msg = input("user: ")
            ans = self.generate_answer(msg)
            print()
            print("answer")
            print(ans)


if __name__ == "__main__":
    vectordb = VectorDB(upload=False)
    lc = LangChain(db=vectordb)
    lc.TEST()
