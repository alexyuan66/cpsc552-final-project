GENERATE_INITIAL_RESPONSE_PROMPT = """We provide you with a question related to software engineering, and a number of questions,
tags, and answers from Stack Overflow that may be relevant. Provide a detailed, informative answer to the following software engineering
related question while referencing the Stack Overflow responses if relevant.

Question:
{question}

List of Relevant Stack Overflow Questions and Answers:
{rag_data}
"""


GENERATE_FEEDBACK_PROMPT = """You are assisting in improving responses to questions related to software engineering.
Given the following question and an initial draft answer, generate up to {max_feedback} concise feedback points describing
how the draft can be improved. You are also given a number of questions, tags, and answers from Stack Overflow that may be
relevant. Focus on factual correctness, completeness, and clarity. If the given sources from Stack Overflow are relevant, also
check whether the response references them.
    
Question:
{question}
    
Initial Response:
{initial_response}

List of Relevant Stack Overflow Questions and Answers:
{rag_data}
    
Feedback (list up to {max_feedback} feedback as numbered points):"""


GENERATE_REFINED_RESPONSE_PROMPT = """You are assisting in improving responses to questions related to software engineering.
We provide you with a question related to software engineering, an initial draft answer, a list of feedback on the initial
answer, and a number of questions, tags, and answers from Stack Overflow that may be relevant to the question. Please incorporate
the feedback to improve the draft answer. Only modify the parts that require enhancement as noted in the feedback, keeping the
other sentences unchanged. Do not omit any crucial information from the original answer unless the feedback specifies that certain
sentences are incorrect and should be removed. If you add new paragraphs or discussions, ensure that you are not introducing repetitive
content or duplicating ideas already included in the original response.

Question:
{question}

Draft Answer:
{initial_response}

List of Feedback Points:
{feedback}

List of Relevant Stack Overflow Questions and Answers:
{rag_data}
"""