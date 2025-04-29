from collections.abc import Callable

from model.prompts import GENERATE_INITIAL_RESPONSE_PROMPT, GENERATE_FEEDBACK_PROMPT, GENERATE_REFINED_RESPONSE_PROMPT

from model.rag import RAG


class OpenCoder:
    def __init__(self, pipeline: Callable):
        """
        Args:
            pipeline (function): A LLM model's prompting function that inputs a prompt (str) and outputs
                                 a response (str).
        """
        self.pipeline = pipeline
        self.rag = RAG()
    

    def generate(self, query: str, max_feedback=5):
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
        rag_data = self.rag.retrieve(query)

        # 2) Generate initial response using both the query and RAG data
        prompt = GENERATE_INITIAL_RESPONSE_PROMPT.format(**{
            'question': query,
            'rag_data': rag_data
        })
        initial_response = self.pipeline(prompt)

        # 3) Generate feedback on the initial response
        prompt = GENERATE_FEEDBACK_PROMPT.format(**{
            'max_feedback': max_feedback,
            'question': query,
            'initial_response': initial_response,
            'rag_data': rag_data
        })
        feedback = self.pipeline(prompt)

        # 4) Refine response based on feedback
        prompt = GENERATE_REFINED_RESPONSE_PROMPT.format(**{
            'question': query,
            'initial_response': initial_response,
            'feedback': feedback,
            'rag_data': rag_data
        })
        final_response = self.pipeline(prompt)

        return final_response

