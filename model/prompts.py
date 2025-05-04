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

# CoT-enhanced versions
COT_GENERATE_INITIAL_RESPONSE_PROMPT = """
You are an expert software‑engineering mentor.  
Follow the two‑phase approach below:

**Phase 1 – Silent reasoning (chain‑of‑thought – DO NOT output):**  
• Break the problem into sub‑questions.  
• Retrieve any concepts, code idioms, or Stack Overflow nuggets that help.  
• Decide the best order to present the information.  

**Phase 2 – Final answer (visible to user):**  
Write a clear, thorough reply to the user’s software‑engineering question, weaving in the relevant Stack Overflow material where appropriate.  
Use code blocks, numbered steps, or short paragraphs for readability.  
Do **not** reveal your hidden reasoning.

---
**Question**  
{question}

---
**Relevant Stack Overflow Excerpts**  
{rag_data}
"""

COT_GENERATE_FEEDBACK_PROMPT = """
You are a meticulous answer reviewer.  
First think step‑by‑step (privately) about weaknesses in the draft, then output up to {max_feedback} short, actionable feedback bullets.

**Silent chain‑of‑thought (do NOT reveal):**  
• Compare the draft with the Stack Overflow sources.  
• Check factual accuracy, completeness, clarity, citation of sources, and code quality.  
• Decide which {max_feedback} issues are most important.

**Visible output format:**  
1. <feedback point #1>  
2. <feedback point #2>  
…

---
**Question**  
{question}

---
**Draft Answer**  
{initial_response}

---
**Relevant Stack Overflow Excerpts**  
{rag_data}
"""

COT_GENERATE_REFINED_RESPONSE_PROMPT = """
Act as an answer‑improvement assistant.  
Proceed in two phases:

**Phase 1 – Internal reasoning (hidden):**  
• For each feedback bullet, locate the specific sentence or section it targets.  
• Plan concise edits or additions that fix the issue without disturbing correct parts.  

**Phase 2 – Revised answer (shown to user):**  
Apply only the planned changes.  
Keep untouched sentences identical to the draft.  
Avoid new redundancies and ensure the final answer references Stack Overflow material when useful.

---
**Question**  
{question}

---
**Original Draft**  
{initial_response}

---
**Feedback List**  
{feedback}

---
**Relevant Stack Overflow Excerpts**  
{rag_data}
"""

RERANKER_GENERATE_BETTER_RESPONSE_PROMPT = """
You are given:

1. A user query.
2. Two generated responses: Response A and Response B.
3. Relevant retrieved context from a knowledge base (RAG data).

Your task is to evaluate both responses in light of the retrieved context and determine which one is better at answering the user query.

Please follow these guidelines:
- Consider factual accuracy: Which response aligns better with the retrieved context?
- Consider relevance: Which response addresses the user's query more directly?
- Consider clarity and completeness: Which response explains the answer more thoroughly and understandably?

Now, evaluate the following:

User Query:
{query}

Retrieved Context:
{rag_data}

Response A:
{response_a}

Response B:
{response_b}

Your Evaluation:
Which response is better? Please respond with \boxed{answer}, where answer is either A or B.
"""

RERANKER_GENERATE_BETTER_REFINED_PROMPT = """
You are given:

1. A user query.
2. An initial generated response
3. Feedback on the response
4. Relevant retrieved context from a knowledge base (RAG data).
5. Two sets of refined responses generated based on the feedback

Your task is to evaluate both refined responses in light of the retrieved context, initial response, and feedback and determine which refined response better answers the user query and takes the feedback into account.

Please follow these guidelines:
- Feedback incorporation: Which response more effectively addresses the specific feedback?
- Context alignment: Which response better reflects the facts in the retrieved context?
- Query relevance: Which response more directly and completely answers the user query?
- Clarity and fluency: Which response is clearer and more well-structured?

Now, evaluate the following:

User Query:
{query}

Initial Response:
{initial_response}

Feedback:
{feedback}

Retrieved Context:
{rag_data}

Refined Response A:
{response_a}

Refined Response B:
{response_b}

Your Evaluation:
Which refined response is better? Please respond with \boxed{answer}, where answer is either A or B.
"""

