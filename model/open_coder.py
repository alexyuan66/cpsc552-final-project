from collections.abc import Callable

from model.prompts import *

from model.rag import RAG
import re


from collections import defaultdict

class SafeDict(defaultdict):
    def __missing__(self, key):
        return f"{{{key}}}"

def escape_curly_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

def delete_curly_braces(text: str) -> str:
    return text.replace("{", "left curly brace").replace("}", "right curly brace")

class OpenCoder:
    def __init__(self, pipeline: Callable, limit=None):
        """
        Args:
            pipeline (function): A LLM model's prompting function that inputs a prompt (str) and outputs
                                 a response (str).
        """
        self.pipeline = pipeline
        self.rag = RAG(limit=limit)

    def __call__(self, queries, max_feedback=5, use_cot: bool = False, rerank_initial: bool = False, rerank_refined: bool = False, use_naive: bool = False):
        # Accept str or list[str] so run_evaluation can pass a batch
        if isinstance(queries, str):
            return self._generate_one(queries, max_feedback)
        return self._generate_batch(list(map(delete_curly_braces, queries)), max_feedback, use_cot, rerank_initial, rerank_refined, use_naive)

    # ask model to generate 2 repsonses and decide which among them is better
    def _rerank_2(self, prompts, rerank_prompt_templates):
        rerank_prompt_templates_clean = rerank_prompt_templates
        responses = []
        for i in range(2):
            out = self.pipeline(prompts)
            out_arr = [x[0]["generated_text"] for x in out]
            responses.append(out_arr)

        safe_values = lambda ra, rb: SafeDict(str, {
            "response_a": delete_curly_braces(ra),
            "response_b": delete_curly_braces(rb),
        })
        rerank_prompts = [
            s.format_map(safe_values(ra,rb))
            for s, ra, rb in zip(rerank_prompt_templates_clean, responses[0], responses[1])
        ]
        reranked_out = self.pipeline(rerank_prompts)
        better_responses_ind = [x[0]["generated_text"] for x in reranked_out]
        # prompted to output A or B for which response is better
        final_out = []
        for i, better in enumerate(better_responses_ind):
            match = re.search(r'\\boxed\{(A|B)\}', better)
            if match:
                letter = match.group(1)
                if letter == 'A':
                    final_out.append(responses[0][i])
                elif letter == 'B':
                    final_out.append(responses[1][i])
                else:
                    # print(f"ERROR: Reranker response \\boxed{{{letter}}} does not match expected format of A or B")
                    final_out.append(responses[0][i])
            else:
                # print("Neither \\boxed{A} nor \\boxed{B} found")
                # print(f"LLM out: {better}")
                final_out.append(responses[0][i])

        return final_out

    # -------- NEW: fully GPU‑batched generation --------
    def _generate_batch(self, queries, max_feedback=5, use_cot: bool = False, rerank_initial: bool = False, rerank_refined: bool = False, use_naive: bool = False):
        # 1 · RAG retrieval for every question
        rag_data = self.rag.retrieve_batch(queries) if not use_naive else self.rag.retrieve_batch_naive(queries)
        rag_data = [delete_curly_braces(r) for r in rag_data]

        # 2 · Initial answers (single GPU call)
        init_prompts = [
            (COT_GENERATE_INITIAL_RESPONSE_PROMPT if use_cot else GENERATE_INITIAL_RESPONSE_PROMPT).format(
                question=q, rag_data=r)
            for q, r in zip(queries, rag_data)
        ]
        if rerank_initial:
            safe_values = lambda q, r: SafeDict(str, {
                "query": q,
                "rag_data": escape_curly_braces(r),
            })
            rerank_prompt_templates = [
                RERANKER_GENERATE_BETTER_RESPONSE_PROMPT.format_map(safe_values(q,r))
                for q, r in zip(queries, rag_data)
            ]

            initial = self._rerank_2(init_prompts, rerank_prompt_templates)

        else:
            init_out = self.pipeline(init_prompts)
            initial = [x[0]["generated_text"] for x in init_out]

        # 3 · Feedback (single GPU call)
        fb_prompts = [
            (COT_GENERATE_FEEDBACK_PROMPT if use_cot else GENERATE_FEEDBACK_PROMPT).format(
                max_feedback=max_feedback, question=q,
                initial_response=ir, rag_data=r)
            for q, ir, r in zip(queries, initial, rag_data)
        ]
        fb_out = self.pipeline(fb_prompts)
        feedback = [x[0]["generated_text"] for x in fb_out]

        # 4 · Refinement (single GPU call)
        ref_prompts = [
            (COT_GENERATE_REFINED_RESPONSE_PROMPT if use_cot else GENERATE_REFINED_RESPONSE_PROMPT).format(
                question=q, initial_response=ir,
                feedback=fb, rag_data=r)
            for q, ir, fb, r in zip(queries, initial, feedback, rag_data)
        ]
        if rerank_refined:
            safe_values = lambda q, r, ir, fb: SafeDict(str, {
                "query": q,
                "rag_data": escape_curly_braces(r),
                "initial_response": delete_curly_braces(ir),
                "feedback": delete_curly_braces(fb),
            })
            rerank_prompts = [
                RERANKER_GENERATE_BETTER_REFINED_PROMPT.format_map(safe_values(q, r, ir, fb))
                for q, r, ir, fb in zip(queries, rag_data, initial, feedback)
            ]
            ref_out_final = self._rerank_2(ref_prompts, rerank_prompts)
        else:   
            ref_out = self.pipeline(ref_prompts)
            ref_out_final = [x[0]["generated_text"] for x in ref_out]
        return ref_out_final

    def _generate_one(self, query: str, max_feedback=5):
        """
        Generate a response to a query using the OpenCoder framework, which adapts the OpenScholar
        framework (https://arxiv.org/pdf/2411.14199) to specialize in answering questions in the field
        of software engineering.

        Args:
            query (str): The query to answer.
            max_feedback (int): Maximum number of feedbacks to be generated and used during response refinement.

        Returns:
            str: A response to the query generated using the OpenCoder framework.
        """
        # 1) Retrieve most relevant Stack Overflow QA points
        rag_data = self.rag.retrieve_batch(query)if not self.use_naive else self.rag.retrieve_batch_naive(query)

        # 2) Generate initial response using both the query and RAG data
        if self.use_cot:
            prompt = COT_GENERATE_INITIAL_RESPONSE_PROMPT.format(**{
            'question': query,
            'rag_data': rag_data
        })
        else:
            prompt = GENERATE_INITIAL_RESPONSE_PROMPT.format(**{
                'question': query,
                'rag_data': rag_data
            })
        initial_response = self.pipeline(prompt)

        # 3) Generate feedback on the initial response
        if self.use_cot:
            prompt = COT_GENERATE_FEEDBACK_PROMPT.format(**{
                'max_feedback': max_feedback,
                'question': query,
                'initial_response': initial_response,
                'rag_data': rag_data
            })
        else:
            prompt = GENERATE_FEEDBACK_PROMPT.format(**{
                'max_feedback': max_feedback,
                'question': query,
                'initial_response': initial_response,
                'rag_data': rag_data
            })
        feedback = self.pipeline(prompt)

        # 4) Refine response based on feedback
        if self.use_cot:
            prompt = COT_GENERATE_REFINED_RESPONSE_PROMPT.format(**{
                'question': query,
                'initial_response': initial_response,
                'feedback': feedback,
                'rag_data': rag_data
            })
        else:
            prompt = GENERATE_REFINED_RESPONSE_PROMPT.format(**{
                'question': query,
                'initial_response': initial_response,
                'feedback': feedback,
                'rag_data': rag_data
            })
        final_response = self.pipeline(prompt)

        return final_response

