# coding:utf-8

"""
기존 GPT Prompt Engineer와 유사한 구조로 작성했습니다.
정확한 정답이 정해지지 않은 경우, 해당 파일을 활용하실 수 있습니다.
라이브러리 형태로 코드를 구성합니다.
"""

import ranking_util
from ranking_operations import GptRankingGenerate, GptRankingCompare



class GPTRanking:
    def __init__(self,
                 prompt_chain,
                 description,
                 test_cases,
                 user_prompt_list=None,
                 use_rar=False,
                 use_csv=False,
                 ranking_model_name="gpt-3.5-turbo",
                 judge_prompt=None,
                 n=5
                 ):
        """
        GPTRanking 클래스의 생성자입니다.
        prompt_chain에 의해서 생성되는 결과는 문자열로 반환이 되어야 됩니다.

        :param prompt_chain: 프롬프트 체인을 나타내는 LangChain 객체입니다.
        :param description: 프롬프트 생성을 위한 작업 또는 문맥을 설명하는 문자열입니다.
        :param test_cases: 프롬프트를 생성하는 데 사용되는 다양한 시나리오나 입력을 나타내는 문자열 목록입니다.
        :param user_prompt_list: (선택 사항) 비교에 포함할 사용자 정의 프롬프트의 목록입니다.
        :param use_rar: (선택 사항) 프롬프트 생성에 RAR 기법을 사용할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
        :param use_csv: (선택 사항) 프롬프트 평가를 CSV 파일에 저장할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
        :param ranking_model_name: (선택 사항) 기본값은 "gpt-3.5-turbo" 입니다.
        :param judge_prompt: (선택 사항) 기본값은 None 입니다.
        :param n: (선택 사항) 생성할 프롬프트의 수를 지정하는 정수입니다. 기본값은 5입니다.

        """
        self.prompt_chain = prompt_chain
        self.description = description
        self.test_cases = test_cases
        self.user_prompt_list = user_prompt_list
        self.use_rar = use_rar
        self.use_csv = use_csv
        self.ranking_model_name = ranking_model_name
        self.judge_prompt = judge_prompt
        self.n = n

    def generate_prompts(self):
        """
        주어진 설명과 테스트 케이스를 기반으로 프롬프트를 생성합니다.

        :return: gen_prompt_list: 생성된 프롬프트의 목록입니다.
        """
        gen_prompt_list = GptRankingGenerate(description=self.description,
                                              test_cases=self.test_cases,
                                              use_rar=self.use_rar,
                                              n=self.n).make()
        return gen_prompt_list

    def compare_prompts(self, compare_prompt, **prompt_kwargs):
        """
        주어진 설명, 테스트 케이스 및 비교할 프롬프트 목록을 기반으로 프롬프트를 비교합니다.

        :param compare_prompt: 비교할 프롬프트 목록입니다.
        :param (선택 사항) judge_prompt: 평가에 필요한 기준을 추가합니다. 기본값은 None입니다.
        :param (선택 사항) prompt_kwargs: 프롬프트 생성 및 비교 함수에 전달할 추가 키워드 인수입니다.
        :return: prompt_ratings: 생성된 프롬프트의 평가를 담은 사전입니다.
        """
        prompt_ratings = GptRankingCompare(description=self.description,
                                            compare_prompt=compare_prompt,
                                            test_cases=self.test_cases,
                                            llm_chain=self.prompt_chain
                                           ).make(judge_prompt=self.judge_prompt,
                                                  ranking_model_name=self.ranking_model_name,
                                                  **prompt_kwargs)
        return prompt_ratings

    def generate_and_compare_prompts(self, **prompt_kwargs):
        """
        주어진 설명과 테스트 케이스를 기반으로 프롬프트를 생성하고 비교합니다.

        :param prompt_kwargs: 프롬프트 생성 및 비교 함수에 전달할 추가 키워드 인수입니다.
        :param (선택 사항) judge_prompt: 평가에 필요한 기준을 추가합니다. 기본값은 None입니다.
        :return: prompt_ratings: 생성된 프롬프트의 평가를 담은 사전입니다.
        """
        gen_prompt_list = self.generate_prompts()

        if self.user_prompt_list is None:
            compare_prompt = gen_prompt_list
        else:
            compare_prompt = self.user_prompt_list + gen_prompt_list

        prompt_ratings = self.compare_prompts(compare_prompt, **prompt_kwargs)

        if self.use_csv:
            ranking_util.make_csv(prompt_ratings)

        return prompt_ratings