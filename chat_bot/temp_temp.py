import os
import json
import openai

from database import VectorDB

from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI

INTENT_PROMPT_TEMPLATE = os.path.join("./prompt_template", "parse_intent.txt")
DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join("./prompt_template", "default_response.txt")
QUESTION_RESPONSE_PROMPT_TEMPLATE = os.path.join("./prompt_template", "question_response.txt")

def read_prompt_template(file_path: str) -> str:
        with open(file_path, "r") as f:
            prompt_template = f.read()

        return prompt_template
    
def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

class LangChain():
    def __init__(self, db=None, api_key = os.environ["OPENAI_API_KEY"]):
        self.db = db
        self.openai = openai
        self.openai.api_key = api_key
        self.llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        self.chain = None

        self.parse_intent_chain = create_chain(
            llm=self.llm,
            template_path=INTENT_PROMPT_TEMPLATE,
            output_key="intent",
        )
        self.default_chain = create_chain(
            llm=self.llm,
            template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE,
            output_key="output",
        )
        self.question_chain = create_chain(
            llm=self.llm,
            template_path=QUESTION_RESPONSE_PROMPT_TEMPLATE,
            output_key="output",
        )

    
    def generate_answer(self, user_message):
        context = dict(user_message=user_message)
        context["input"] = context["user_message"]
        context["intent_list"] = read_prompt_template(INTENT_PROMPT_TEMPLATE)
        context["chat_history"] = ""
        intent = self.parse_intent_chain.run(context)

        if intent == "greeting":
            context["message"] = user_message
            answer = self.default_chain.run(context)
        else:
            context["related_documents"] = self.db.get_data_from_db(user_message)
            context["compressed_web_search_results"] = ""


            print("related docu")
            print(context["related_documents"])
            answer = self.question_chain.run(context)

        return answer


    def TEST(self):
        msg = "카카오 소셜에 대해서 궁금해"
        ans = self.generate_answer(msg)
        print()
        print("answer")
        print(ans)


    def funccall(self):
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
    


if __name__ == "__main__":
    vectordb = VectorDB(upload=False)
    lc = LangChain(db=vectordb)
    lc.TEST()
