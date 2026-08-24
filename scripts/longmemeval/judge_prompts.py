"""
LongMemEval answer-correctness judge prompts, ported verbatim from the
official benchmark's evaluation harness (src/evaluation/evaluate_qa.py,
https://github.com/xiaowu0162/LongMemEval, get_anscheck_prompt()) so Aeon's
scores are comparable to published LongMemEval numbers -- this is
deliberately NOT a house style rewrite.

The official script runs this prompt through GPT-4o/GPT-4o-mini/a hosted
Llama-3.1-70B; this benchmark stage runs it through a local Ollama model
instead (v4-plan.md Stage 6), since a from-scratch hosted-API integration
was explicitly out of scope for this pass -- see v4-plan.md for the
model-choice rationale and its accuracy caveat vs. the GPT-4o-judged
published baselines.
"""


def get_anscheck_prompt(
    question_type: str, question: str, answer: str, response: str,
    abstention: bool = False,
) -> str:
    """Builds the yes/no correctness-judge prompt for one QA pair.

    Args:
        question_type: One of 'single-session-user', 'single-session-
            assistant', 'multi-session', 'temporal-reasoning',
            'knowledge-update', 'single-session-preference'.
        question: The benchmark question text.
        answer: The reference answer (or, for single-session-preference,
            the rubric text).
        response: The system-under-test's generated answer (the
            "hypothesis").
        abstention: True for questions whose id contains '_abs' (the
            official benchmark's unanswerable-question augmentation) --
            switches to the identify-as-unanswerable judge prompt.

    Returns:
        A single-turn prompt string to send to the judge model at
        temperature 0, expecting a short "yes"/"no" completion.
    """
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and "
            "a response from a model. Please answer yes if the model "
            "correctly identifies the question as unanswerable. The model "
            "could say that the information is incomplete, or some other "
            "information is given but the asked information is not.\n\n"
            "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as "
            "unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)

    if question_type in ("single-session-user", "single-session-assistant", "multi-session"):
        template = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the "
            "intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the "
            "information required by the answer, answer no. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)

    if question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response is "
            "equivalent to the correct answer or contains all the "
            "intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the "
            "information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days. If "
            "the question asks for the number of days/weeks/months, etc., "
            "and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still "
            "correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
        return template.format(question, answer, response)

    if question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response "
            "from a model. Please answer yes if the response contains the "
            "correct answer. Otherwise, answer no. If the response "
            "contains some previous information along with an updated "
            "answer, the response should be considered as correct as long "
            "as the updated answer is the required answer.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)

    if question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if "
            "the response satisfies the desired response. Otherwise, "
            "answer no. The model does not need to reflect all the points "
            "in the rubric. The response is correct as long as it recalls "
            "and utilizes the user's personal information correctly.\n\n"
            "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the "
            "model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)

    raise NotImplementedError(f"Unknown LongMemEval question_type: {question_type!r}")
