# Langchain Ranking

https://github.com/mshumer/gpt-prompt-engineer

해당 레포지토리기반으로 랭체인을 통해 구현한 프로젝트입니다. 
gpt-prompt-engineer의 아이디어는 대단합니다. 
저는 Langchain을 통해 LLM Chain만 생성 할 수 있으면 모든 모델을 GPT모델들을 통해 평가를 할 수 있는 랭킹 프로젝트를 생성하였습니다. 

# 작동 방법
```
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
```
# 실행 결과
```commandline
['Discover the top 10 secrets to unlocking your full potential.', "Craft an enticing headline that captivates your audience's attention.", 'Craft compelling headlines that entice and engage your target audience.']

11%|███▊                              | 2/18 [00:09<01:15,  4.73s/it]
 Winner: Discover the top 10 secrets to unlocking your full potential.
17%|█████▋                            | 3/18 [00:11<00:52,  3.49s/it]
 Draw
22%|███████▌                          | 4/18 [00:14<00:48,  3.44s/it]
Winner: Discover the top 10 secrets to unlocking your full potential.
28%|█████████▍                        | 5/18 [00:17<00:43,  3.35s/it]
Winner: example1
 33%|███████████▎                      | 6/18 [00:20<00:38,  3.19s/it
Winner: Craft an enticing headline that captivates your audience's attention.
39%|█████████████▏                    | 7/18 [00:22<00:31,  2.88s/it]
Winner: Craft an enticing headline that captivates your audience's attention.
 44%|███████████████                   | 8/18 [00:27<00:34,  3.44s/it]
Winner: Craft compelling headlines that entice and engage your target audience.
Winner: Craft compelling headlines that entice and engage your target audience.
 56%|██████████████████▎              | 10/18 [00:31<00:21,  2.66s/it]
 Winner: Craft compelling headlines that entice and engage your target audience.
 61%|████████████████████▏            | 11/18 [00:33<00:16,  2.42s/it]
 Draw
 67%|██████████████████████           | 12/18 [00:35<00:13,  2.22s/it]
 Winner: Discover the top 10 secrets to unlocking your full potential.
Draw
 78%|█████████████████████████▋       | 14/18 [00:39<00:08,  2.20s/it]
 Draw
 83%|███████████████████████████▌     | 15/18 [00:41<00:06,  2.22s/it]
 Draw
Winner: Discover the top 10 secrets to unlocking your full potential.
 94%|███████████████████████████████▏ | 17/18 [00:45<00:02,  2.16s/it]
 Draw
100%|█████████████████████████████████| 18/18 [00:47<00:00,  2.13s/it]
Draw
100%|█████████████████████████████████| 18/18 [00:49<00:00,  2.76s/it]
Draw
{'example1': 1118.6540526700558, 'Discover the top 10 secrets to unlocking your full potential.': 1257.36515660952, "Craft an enticing headline that captivates your audience's attention.": 1203.8262872651787, 'Craft compelling headlines that entice and engage your target audience.': 1220.1545034552455}

```
