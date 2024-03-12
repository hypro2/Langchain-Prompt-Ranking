from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from gpt_ranking import GPTRanking

# 사용 방법
# custom llm chain 정의
prompt = ChatPromptTemplate.from_messages([("system", "Answer in {A}"), ("user", "{input}")])
result_model = ChatOpenAI(model_name="gpt-3.5-turbo")
custom_chain = prompt | result_model | StrOutputParser()

# 하고 싶은 작업에 대한 설명을 입력합니다.
description1 = "Create a landing page headline."

# 결과 값으로 받고 싶은 예시를 몇 가지 입력합니다.(적당한 개수가 있어야 합니다.)
test_cases1 = [
    "Promote your innovative new fitness app smartly",
    "Why a vegan diet is good for your health",
    "Introducing a new online course on digital marketing.",
]

# 직접 작성한 프롬프트의 성능도 비교하려면 여기에 입력합니다. 입력하지 않아도 됩니다.
user_prompt_list1 = ["example1"]

gpt_rank = GPTRanking(prompt_chain=custom_chain,
                      description=description1,
                      test_cases=test_cases1,
                      user_prompt_list=user_prompt_list1,
                      ranking_model_name="gpt-3.5-turbo",
                      use_rar=False,
                      use_csv=False,
                      n=3,
                      )

gpt_rank.generate_and_compare_prompts(A="Korean")
