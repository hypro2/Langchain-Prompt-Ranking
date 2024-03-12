# coding:utf-8
import pandas as pd
from prompt_variable import PromptVariable as pv

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

load_dotenv()


def make_csv(prompt_ratings):
    try:
        # ELO rating 결과를 출력한다.
        result_list = []
        for prompt, rating in sorted(prompt_ratings.items(), key=lambda item: item[1], reverse=True):
            result_list.append([prompt, rating])

        df = pd.DataFrame(result_list, columns=["Prompt", "Rating"])
        df.to_csv('./prompt_rating.csv', index=False)

        print(result_list)

    except Exception as e:
        print(e)
        raise e


def prompt_template(system_message: str, ai_message: str = ""):
    prompt = ChatPromptTemplate.from_messages(
        [("system", f"{system_message}"),
         ("ai", f"{ai_message}"),
         ("user", "{input}")]
    )

    return prompt


def rar_prompting(input_prompt: str,
                  model_name: str = "gpt-3.5-turbo"):
    RaR_prompt = pv.RaR_PROMPT
    prompt = prompt_template(RaR_prompt)

    model = ChatOpenAI(model_name=model_name,
                       temperature=0.9,
                       max_retries=40)

    chain = prompt | model | StrOutputParser()

    rar_result = chain.invoke({"input": f"{input_prompt}"})

    try:
        rar_result = rar_result.split(":")[1]
    except:
        pass
    rar_result = rar_result.replace("Rewritten Prompt:", "")
    rar_result = rar_result.replace("Rephrased Prompt:", "")

    return rar_result.strip()


def generate_prompts(user_message: str,
                     system_message: str = None,
                     ai_message: str = None,
                     model_name: str = "gpt-3.5-turbo",
                     n: int = 10):
    if system_message is None:
        system_message = pv.GPT_GENERATE_SYSTEM

    if ai_message is None:
        ai_message = ""

    prompt = prompt_template(system_message, ai_message)

    model = ChatOpenAI(model_name=model_name,
                       temperature=0.9,
                       max_retries=40)

    chain = prompt | model | StrOutputParser()

    gen_prompts_list = []

    for _ in range(n):
        gen_prompt = chain.invoke({"input": f"{user_message}"})
        RaR_result = rar_prompting(gen_prompt)
        gen_prompts_list.append(RaR_result)

    return gen_prompts_list


def get_score(description: str,
              example_cases: list,
              prompt_result1: str,
              prompt_result2: str,
              judge_prompt: str = None,
              ai_message: str = None,
              model_name: str = "gpt-3.5-turbo"):
    if judge_prompt is None:
        ranking_prompt = pv.GPT_RANK_SYSTEM
    else:
        ranking_prompt = f"{pv.GPT_RANK_SYSTEM}\nJudgment criteria:\n{judge_prompt}"

    if ai_message is None:
        ai_message = ""

    prompt = prompt_template(ranking_prompt, ai_message)

    model = ChatOpenAI(model_name=model_name,
                       temperature=0.5,
                       max_retries=40,
                       max_tokens=1,
                       model_kwargs={'logit_bias': {'32': 100, '33': 100}})

    chain = prompt | model | StrOutputParser()

    score = chain.invoke({"input": f"Task: {description}"
                                   f"Generation A: {prompt_result1}"
                                   f"Generation B: {prompt_result2}"
                                   f"Example cases: {example_cases}"})

    return score


def get_score_with_label(prompt_result1: str,
                         prompt_result2: str,
                         label: str,
                         ranking_prompt: str = None,
                         ai_message: str = None,
                         model_name: str = "gpt-3.5-turbo"):
    if ranking_prompt is None:
        ranking_prompt = pv.GPT_RANK_LABEL_SYSTEM

    if ai_message is None:
        ai_message = ""

    prompt = prompt_template(ranking_prompt, ai_message)

    model = ChatOpenAI(model_name=model_name,
                       temperature=0.5,
                       max_retries=40,
                       max_tokens=1,
                       model_kwargs={'logit_bias': {'32': 100, '33': 100}})

    chain = prompt | model | StrOutputParser()

    score = chain.invoke({"input": f"Label: {label}"
                                   f"Generation A: {prompt_result1}"
                                   f"Generation B: {prompt_result2}"})

    return score


if __name__ == "__main__":
    user_message1 = "hi"
    gen_prompts_list1 = generate_prompts(user_message1)
