import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage


import tiktoken
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

from langchain.utilities import DuckDuckGoSearchAPIWrapper

api_key = os.environ["openai_api_key"]


def langchain_ex2():
    print("example 2")
    system_message = "assistant는 마케팅 문구 작성 도우미로 동작한다. user의 내용을 참고하여 마케팅 문구를 작성해라"
    system_message_prompt = SystemMessage(content=system_message)

    human_template = ("제품 이름: {product_name}\n"
                  "제품 설명: {product_desc}\n"
                  "제품 톤앤매너: {product_tone_and_manner}\n"
                  "위 정보를 참조해서 마케팅 문구 만들어줘"
                  )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat = ChatOpenAI(temperature=0.8)
    print(chat)
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    aa = chain.run(product_name="나이키 신발",
                    product_desc="편안한 착용감",
                    product_tone_and_manner="유쾌")

    print(aa)
    print()
    print()

def langchain_ex3():
    def clean_html(url):
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        return text
    
    def truncate_text(text, max_tokens=3000):
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:  # 토큰 수가 이미 3000 이하라면 전체 텍스트 반환
            return text
        return enc.decode(tokens[:max_tokens])

    def task(search_result):
        title = search_result['title']
        url = search_result['link']
        snippet = search_result['snippet']

        content = clean_html(url)
        full_content = f"제목: {title}\n발췌: {snippet}\n전문: {content}"
        full_content_truncated = truncate_text(full_content, max_tokens=3500)

        ############
        system_message = "assistant는 user의 내용을 bullet point 3줄로 요약하라. 영어인 경우 한국어로 번역해서 요약하라."
        system_message_prompt = SystemMessage(content=system_message)

        human_template = "{text}\n---\n위 내용을 bullet point로 3줄로 한국어로 요약해"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])

        chain = LLMChain(llm=llm, prompt=chat_prompt)
        summary = chain.run(text=full_content_truncated)

        result = {"title": title,
                "url": url,
                "content": content,
                "summary": summary
                }

        return result
    
    llm = ChatOpenAI(temperature=0.8)

    search = DuckDuckGoSearchAPIWrapper()
    search.region = 'kr-kr'
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    topic = "카카오"
    search_result = search.results(topic, max_results=1)[0]

    result = task(search_result)
    print(result)

def langchain_ex4():
    def create_chain(llm, template_path, output_key):
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=read_prompt_template(template_path),
            ),
            output_key=output_key,
            verbose=True,
        )

    def generate_novel(genre, characters, news_text) -> dict[str, str]:
        writer_llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo-16k")

        novel_idea_chain = create_chain(writer_llm, STEP1_PROMPT_TEMPLATE, "novel_idea")
        novel_outline_chain = create_chain(writer_llm, STEP2_PROMPT_TEMPLATE, "novel_outline")
        novel_plot_chain = create_chain(writer_llm, STEP3_PROMPT_TEMPLATE, "novel_plot")
        novel_chapter_chain = create_chain(writer_llm, WRITE_PROMPT_TEMPLATE, "output")

        preprocess_chain = SequentialChain(
            chains=[
                novel_idea_chain,
                novel_outline_chain,
                novel_plot_chain,
            ],
            input_variables=["genre", "characters", "news_text"],
            output_variables=["novel_idea", "novel_outline", "novel_plot"],
            verbose=True,
        )

        context = dict(
            genre=genre,
            characters=characters,
            news_text=news_text
        )
        context = preprocess_chain(context)

        context["novel_chapter"] = []
        for chapter_number in range(1, 3):
            context["chapter_number"] = chapter_number
            context = novel_chapter_chain(context)
            context["novel_chapter"].append(context["output"])

        contents = "\n\n".join(context["novel_chapter"])
        return {"results": contents}


if __name__ == "__main__":
    langchain_ex3()
