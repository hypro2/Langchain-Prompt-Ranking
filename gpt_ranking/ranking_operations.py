# coding:utf-8
import itertools
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser

from gpt_ranking.ranking_util import prompt_template, rar_prompting, get_score
from gpt_ranking.prompt_variable import PromptVariable as pv


class GptRankingGenerate:

    def __init__(self, description, test_cases="", system_message=None, ai_message=None, model_name="gpt-3.5-turbo", n=10, use_rar=False):
        self.description = description
        self.test_cases = test_cases
        self.system_message = system_message
        self.ai_message = ai_message
        self.model_name = model_name
        self.n = n
        self.use_rar = use_rar


    def make(self):

        try:
            system_message = self.system_message
            if system_message is None:
                system_message = pv.GPT_GENERATE_SYSTEM

            ai_message = self.ai_message
            if ai_message is None:
                ai_message = ""

            prompt = prompt_template(system_message, ai_message)

            model = ChatOpenAI(model_name=self.model_name,
                               temperature=0.9,
                               max_retries=40)

            chain = prompt | model | StrOutputParser()

            gen_prompts_list = []
            for _ in range(self.n):
                gen_prompt = chain.invoke({"input": f"Here are some test cases: ```{self.test_cases}```\n"
                                                    f"Here is the desctiption of the use-cases: '{self.description}'\n"
                                                    f"Don't include the contents of the test cases. You need to create a universal prompt."
                                                    f"Respond to the prompt and do nothing else. Be creative."})

                if self.use_rar:
                    gen_prompt = rar_prompting(gen_prompt)

                gen_prompts_list.append(gen_prompt)

            print(gen_prompts_list)
            return gen_prompts_list

        except Exception as e:
            print(e)
            raise e


class GptRankingCompare:

    def __init__(self, description, compare_prompt, test_cases, llm_chain):
        self.description = description
        self.compare_prompt = compare_prompt
        self.test_cases = test_cases
        self.llm_chain = llm_chain

    def make(self, judge_prompt=None, ranking_model_name="gpt-3.5-turbo", **kwargs):

        try:
            # 기본 rating은 1200
            prompt_ratings = {prompt: 1200 for prompt in self.compare_prompt}
            prompt_args = {**kwargs}

            # 총 라운드 계산
            total_rounds = len(self.test_cases) * len(self.compare_prompt) * (len(self.compare_prompt) - 1) // 2
            pbar = tqdm(total=total_rounds, ncols=70)

            generation_output = dict()

            for prompt1, prompt2 in itertools.combinations(self.compare_prompt, 2):
                for case in self.test_cases:
                    pbar.update()

                    #prompt 업데이트
                    prompt1_args = prompt_args.copy()
                    prompt2_args = prompt_args.copy()
                    prompt1_args["case"] = case
                    prompt2_args["case"] = case
                    prompt1_args["input"] = prompt1
                    prompt2_args["input"] = prompt2

                    # 출력 생성1
                    if generation_output.get(prompt1, False):
                        prompt_result1 = generation_output.get(prompt1)
                    else:
                        prompt_result1 = self.llm_chain.invoke(prompt1_args)
                        generation_output[prompt1] = prompt_result1
                        print(prompt_result1)

                    # 출력 생성2
                    if generation_output.get(prompt2, False):
                        prompt_result2 = generation_output.get(prompt2)
                    else:
                        prompt_result2 = self.llm_chain.invoke(prompt2_args)
                        generation_output[prompt2] = prompt_result2
                        print(prompt_result2)

                    # 점수 비교
                    score1 = get_score(self.description, case, prompt_result1, prompt_result2, judge_prompt=judge_prompt, model_name=ranking_model_name)
                    score2 = get_score(self.description, case, prompt_result2, prompt_result1, judge_prompt=judge_prompt, model_name=ranking_model_name)

                    # 점수를 숫자로 변환
                    score1 = 1 if score1 == "A" else 0 if score1 == "B" else 0.5
                    score2 = 1 if score2 == 'B' else 0 if score2 == "A" else 0.5
                    score = (score1 + score2) / 2

                    # ELO rating 방법으로 rating 업데이트
                    rating1, rating2 = prompt_ratings[prompt1], prompt_ratings[prompt2]
                    rating1, rating2 = self._update_elo(rating1, rating2, score)

                    prompt_ratings[prompt1], prompt_ratings[prompt2] = rating1, rating2

                    # 해당 라운드의 결과 출력
                    if score > 0.5:
                        print(f"Winner: {prompt1}")
                    elif score < 0.5:
                        print(f"Winner: {prompt2}")
                    else:
                        print("Draw")

            pbar.close()
            print(prompt_ratings)
            # print(generation_output)
            return prompt_ratings

        except Exception as e:
            print(e)
            raise e

    def _update_elo(self, rating1, rating2, score):
        K = 32
        elo1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        elo2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        return rating1 + K * (score - elo1), rating2 + K * ((1 - score) - elo2)
