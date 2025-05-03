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





# Prompt 1 ── initial answer generation, now with explicit CoT guidance
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

# Prompt 2 ── feedback generation with CoT; variable renamed as requested
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

# Prompt 3 ── answer refinement with CoT guidance
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
