from collections.abc import Callable

from prompts import *

from rag import RAG


class OpenCoder:
    def __init__(self, pipeline: Callable, use_cot: bool = False):
        """
        Args:
            pipeline (function): A LLM model's prompting function that inputs a prompt (str) and outputs
                                 a response (str).
        """
        self.pipeline = pipeline
        self.rag = RAG()
        self.use_cot = use_cot

    def __call__(self, queries, max_feedback=5):
        # Accept str or list[str] so run_evaluation can pass a batch
        if isinstance(queries, str):
            return self._generate_one(queries, max_feedback)
        return self._generate_batch(queries, max_feedback)

    # -------- NEW: fully GPU‑batched generation --------
    def _generate_batch(self, queries, max_feedback=5):
        # 1 · RAG retrieval for every question
        rag_data = self.rag.retrieve_batch(queries)

        # 2 · Initial answers (single GPU call)
        init_prompts = [
            (COT_GENERATE_INITIAL_RESPONSE_PROMPT if self.use_cot else GENERATE_INITIAL_RESPONSE_PROMPT).format(
                question=q, rag_data=r)
            for q, r in zip(queries, rag_data)
        ]
        init_out = self.pipeline(init_prompts)
        initial = [x[0]["generated_text"] for x in init_out]

        # 3 · Feedback (single GPU call)
        fb_prompts = [
            (COT_GENERATE_FEEDBACK_PROMPT if self.use_cot else GENERATE_FEEDBACK_PROMPT).format(
                max_feedback=max_feedback, question=q,
                initial_response=ir, rag_data=r)
            for q, ir, r in zip(queries, initial, rag_data)
        ]
        fb_out = self.pipeline(fb_prompts)
        feedback = [x[0]["generated_text"] for x in fb_out]

        # 4 · Refinement (single GPU call)
        ref_prompts = [
            (COT_GENERATE_REFINED_RESPONSE_PROMPT if self.use_cot else GENERATE_REFINED_RESPONSE_PROMPT).format(
                question=q, initial_response=ir,
                feedback=fb, rag_data=r)
            for q, ir, fb, r in zip(queries, initial, feedback, rag_data)
        ]
        ref_out = self.pipeline(ref_prompts)
        return [x[0]["generated_text"] for x in ref_out]

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
        rag_data = self.rag.retrieve_batch(query)

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

