class PromptVariable:
    GPT_EXTRACT_SYSTEM = """You are an AI that creatively generates sentences.
    Generate a sentence that asks you to extract paragraphs related to the {type} of invention.
    The sentences should be in English. For best results, be creative.
    You will be graded on the performance of your sentences, but don't cheat!
    Don't include examples in your sentence. The most important thing is that you only output sentences. Don't include anything else in the message."""
    # """귀하는 문장을 창의적으로 생성하는 인공지능입니다.
    # 발명의 {type}과 관련된 단락을 추출하라는 문장을 생성하세요.
    # 문장은 영어로 작성해야 합니다. 최상의 결과를 얻으려면 창의력을 발휘하세요.
    # 문장의 성능에 따라 채점되지만 속임수를 쓰지 마세요!
    # 문장에 예시를 포함하지 마세요. 가장 중요한 것은 문장만 출력하는 것입니다. 메시지에 다른 내용을 포함하지 마세요."""

    GPT_GENERATE_SYSTEM = """You are an artificial intelligence that creatively generates sentences.
    The sentences must be written in English. Be creative to get the best results.
    You'll be graded on your sentence performance, but don't cheat!
    Don't include example sentences in your message. You're creating a universal prompt candidate. Don't include anything else in the message."""
    # """귀하는 창의적으로 문장을 생성하는 인공지능입니다.
    # 문장은 영어로 작성해야 합니다. 최상의 결과를 얻으려면 창의력을 발휘하세요.
    # 문장 성능에 따라 점수가 매겨지지만 속임수를 쓰지 마세요!
    # 문장에 예문을 포함하지 마세요. 가장 중요한 것은 문장만 출력하는 것입니다. 메시지에 다른 내용을 포함하지 마세요."""

    GPT_RANK_SYSTEM = """Your task is to rank the quality of two outputs generated by different prompts. The prompts are used to generate responses to a given task.
    You will be given two generations, one for the task description, an example case, and one for each system prompt. Rank the generations according to their quality. If generation A is better, respond with 'A'. If generation B is better, respond with a 'B'.
    Remember that to be considered 'better', a generation must be noticeably better than the others, not just good.
    Also, keep in mind that you are your own harshest critic. Only rank one generation as better than another if it is truly impressive.
    Don't include anything else, just respond with a ranking. Be fair and unbiased in your judgment."""
    # """작업은 서로 다른 프롬프트에서 생성된 두 출력물의 품질에 순위를 매기는 것입니다. 프롬프트는 주어진 작업에 대한 응답을 생성하는 데 사용됩니다.
    # 작업 설명, 예시 사례 및 각 시스템 프롬프트에 대해 하나씩 두 세대가 주어집니다. 품질에 따라 세대의 순위를 매겨 보세요. A 세대가 더 좋으면 'A'로 응답합니다. B 세대가 더 좋으면 'B'로 응답하세요.
    # 한 세대가 '더 나은' 것으로 간주되려면 단순히 좋은 것이 아니라 다른 세대보다 눈에 띄게 우수해야 한다는 점을 기억하세요.
    # 또한, 여러분 스스로가 가장 가혹한 비평가라는 점을 명심하세요. 정말 인상적인 경우에만 한 세대를 다른 세대보다 나은 것으로 평가하세요.
    # 다른 내용은 포함하지 말고 순위로만 응답하세요. 공정하고 편견 없이 판단하세요."""

    GPT_RANK_LABEL_SYSTEM = """Your task is to find a generation that is similar to the correct answer.
    You will be given Generation A, Generation B, and the correct answer.
    If Generation A is more similar to the correct answer, you respond with 'A'. If Generation B is more similar to the correct answer, respond with 'B'.
    Don't include anything else, just respond with an 'A' or 'B'. Be fair and unbiased in your judgment."""
    # """당신의 임무는 정답과 비슷한 세대를 찾는 것입니다.
    # A 세대, B 세대, 정답이 주어집니다.
    # A 세대가 정답과 더 유사하면 'A'로 응답합니다. B 세대가 정답과 더 유사하면 'B'로 응답합니다.
    # 다른 내용은 포함하지 말고 'A' 또는 'B'로만 응답하세요. 공정하고 편견 없이 판단하세요."""

    RaR_PROMPT = """ChatGPT, I would like to request your assistance in creating an AI-powered prompt rewriter, which can help me rewrite and refine prompts that I intend to use with you, ChatGPT, for the purpose of obtaining improved responses. To achieve this, I kindly ask you to follow the guidelines and techniques described below in order to ensure the rephrased prompts are more specific, contextual, and easier for you to understand.
    Identify the main subject and objective: Examine the original prompt and identify its primary subject and intended goal. Make sure that the rewritten prompt maintains this focus while providing additional clarity.
    Add context: Enhance the original prompt with relevant background information, historical context, or specific examples, making it easier for you to comprehend the subject matter and provide more accurate responses.
    Ensure specificity: Rewrite the prompt in a way that narrows down the topic or question, so it becomes more precise and targeted. This may involve specifying a particular time frame, location, or a set of conditions that apply to the subject matter.
    Use clear and concise language: Make sure that the rewritten prompt uses simple, unambiguous language to convey the message, avoiding jargon or overly complex vocabulary. This will help you better understand the prompt and deliver more accurate responses.
    Incorporate open-ended questions: If the original prompt contains a yes/no question or a query that may lead to a limited response, consider rephrasing it into an open-ended question that encourages a more comprehensive and informative answer.
    Avoid leading questions: Ensure that the rewritten prompt does not contain any biases or assumptions that may influence your response. Instead, present the question in a neutral manner to allow for a more objective and balanced answer.
    Provide instructions when necessary: If the desired output requires a specific format, style, or structure, include clear and concise instructions within the rewritten prompt to guide you in generating the response accordingly.
    Ensure the prompt length is appropriate: While rewriting, make sure the prompt is neither too short nor too long. A well-crafted prompt should be long enough to provide sufficient context and clarity, yet concise enough to prevent any confusion or loss of focus.
    With these guidelines in mind, I would like you to transform yourself into a prompt rewriter, capable of refining and enhancing any given prompts to ensure they elicit the most accurate, relevant, and comprehensive responses when used with ChatGPT. Rewrite the given sentences."""
    # """ChatGPT, 나는 당신의 AI-powered protectrum을 만드는 당신의 도움을 요청할 것이며, 당신이 사용하는 당신의 의도한 광고에 대한 보고서와 마무리를 작성할 것이다. 당신의 가이드라인과 기술을 따라 Kindly는 당신을 위해 특정한 테마, 컨텍스트, 그리고 당신을 위한 어시스턴트를 리프레시하는 것을 정리하고 조치하는 것에 대해 설명한다.
    # '주요 객체 및 대상 식별' '최초의 프로젝트와 객체 식별'은 골을 의미합니다. 그것은 확실히 이러한 추가적인 제공의 초점에 대한 보상이다.
    # Add context: 당신은 배경 정보, 역사적 배경 정보, 또는 구체적인 샘플, 당신을 위해 객체를 이해하고 더 많은 응답을 제공하기 위해 그것의 원본을 향상시킨다.
    # 구체적인 내용: 주제를 다운받아 당신의 프로젝트를 다시 읽어보십시오. 그것들이 어떻게 정확하고 타겟이 되는지요. 이 5월은 특정 프레임, 위치, 현재의 객체 매트릭스에 적용할 수 있는 조건 세트를 지정하는 것을 포함한다.
    # 한글자막 aurore 수정,배포 자유 / 출처 표시 그것은 분명히 단순한 레위터 프롬프트에서 사용됩니다. 애매모호한 문자 메시지 전달 언어, 또는 완전히 복잡한 어휘 목록을 만드는 것입니다. 이것은 당신이 당신의 목표를 달성하고 정확한 반응을 하도록 도울 것이다.
    # 오픈 엔젠다 통합: 원래의 광고는 포괄적인 연애와 정보에 입각한 공개적인 홍보를 고려하여 현재의 제한된 반응에 대한 질의에 대한 예/아니오를 포함한다.
    # 나는 여기서 묻고 싶다: 당신의 반응에 대한 반응은 느리거나 걱정하지 않는다. 객관적인 사랑과 균형 잡힌 사랑을 위한 중립적인 토론에서 그 문제를 제시한다.
    # 필수적으로 사용할 수 있습니다. 원하는 출력물은 특정 포맷, 스타일, 구조에 대한 요구사항들을 포함하고, 응답성을 생성하기 위한 신속한 안내를 제공한다.
    # 적절한 프롬프트를 표시합니다. WhileRewriting, 반드시 다른 프롬프트를 매핑 또는 짧은 프롬프트 또는 긴 프롬프트에서 실행합니다. Well-crafted 홍보는 충분한 콘텐츠와 안전성을 제공하고, yet를 간결하게 하며, 초점의 혼동을 방지합니다.
    # 당신은 당신의 자신을 당신의 것으로 만들고, 당신은 당신의 것을 당신의 것으로 바꿀 수 있으며, 당신의 것을 당신의 것으로 바꿀 수 있다. GPT 채팅으로 정확하고, 강조되고, 포괄적인 반응을 할 수 있다. '어떻게 하면 당신에게 올 수 있는지' '지시서에 근거한 홍보물에 쓸 수 있는지'"""
