# 🌟 Langchain Ranking 프로젝트 🌟

Langchain 프롬프트 랭킹 프로젝트에 오신 것을 환영합니다! 🎉 이 프로젝트는 gpt-prompt-engineer의 혁신적인 개념을에 감명받아서 랭체인을 통해서 구현한 랭체인을 사용하여 모델을 평가하는 것에 대해 모두 이야기합니다. 🚀

Welcome to the Langchain Prompt Ranking Project!  This project is inspired by the innovative concept of gpt-prompt-engineer and implemented through LangChain, all about evaluating models using LangChain. 

## 🤖 What is gpt-prompt-engineer?

RESPCET : https://github.com/mshumer/gpt-prompt-engineer

'gpt-prompt-engineer'는 GPT-4 및 GPT-3.5-Turbo와 같은 대규모 언어 모델의 성능을 향상시키는 탁월한 도구입니다. 마치 모델을 최적화하는 마법의 지팡이처럼 작동합니다! ✨

'gpt-prompt-engineer' is an excellent tool to improve the performance of large language models such as GPT-4 and GPT-3.5-Turbo. It works like a magic wand to optimize your model! 

## 🛠 How does it work?

'Langchain Ranking'는 특정 사용 사례를 기반으로 다양한 프롬프트를 생성하고 엄격하게 테스트한 후 Elo 등급 시스템을 사용하여 순위를 매깁니다. 💡

'Langchain Ranking' creates a variety of prompts based on specific use cases, tests them rigorously, and ranks them using the Elo rating system. 

## 🏆 Why did you choose Langchain?

Langchain을 사용하면 언어 학습 모델 (LLM) 체인을 구축할 수 있으며, 이는 프로젝트의 기반을 형성합니다. Langchain을 사용하여 모델을 심사하는 데 집중할 수 있습니다. 💪

Langchain allows you to build a chain of language learning models (LLMs), which form the basis of your project. Langchain allows you to focus on vetting your models. 

## 💡 Ranking process

1. **프롬프트 생성:** 🤔 다양한 시나리오에 맞는 다양한 프롬프트를 생성합니다.
2. 
   **Create prompts:**  Create a variety of prompts for different scenarios.
   
3. **테스트 및 비교:** 🧪 각 프롬프트는 엄격한 테스트를 거쳐 성능을 비교합니다.
4. 
   **TEST & COMPARE:**  Each prompt undergoes rigorous testing to compare performance.
   
5. **Elo 등급 시스템:** 📈 프롬프트의 효과에 따라 Elo 등급 시스템을 사용하여 프롬프트를 순위로 매깁니다.
6. 
   **Elo Rating System:**  Use the Elo rating system to rank prompts based on their effectiveness.


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


{
'example1': 1118.6540526700558,

'Discover the top 10 secrets to unlocking your full potential.': 1257.36515660952,

"Craft an enticing headline that captivates your audience's attention.": 1203.8262872651787,

'Craft compelling headlines that entice and engage your target audience.': 1220.1545034552455
}

```
