#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


LIMIT = 3

# Increase CSV field size limit to handle large fields
# Note: must run from pwd = cpsc552-final-project/model
csv.field_size_limit(sys.maxsize)

# Ensure the data_processing directory is on the Python path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, '../data_processing'))

# Import the evaluation function from the data_processing directory
from coderepoqa_evaluation import evaluate_predictions

def load_generation_pipeline(model_name):
    """
    Load a text-generation pipeline for the specified model, handling authentication and device placement.
    Returns a Hugging Face pipeline object that can be called directly on prompts.
    """
    hf_token = os.environ.get("HF_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
    )
    return gen_pipeline


def run_evaluation(csv_path, gen_pipeline):
    """
    Read (question, answer) pairs from csv_path, generate model predictions
    using the provided generation pipeline, and evaluate them via evaluate_predictions.
    """
    questions = []
    ground_truths = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"])
            ground_truths.append(row["answer"])

    predictions = []
    for i, question in enumerate(questions):
        print(f'Answering {i}/{len(questions)}')
        result = gen_pipeline(question)
        pred_answer = result[0]["generated_text"].strip()
        predictions.append(pred_answer)
        if i >= LIMIT:
            break

    metrics = evaluate_predictions(predictions, ground_truths[:LIMIT + 1])
    print(metrics)
    
    print('=======\n\n', predictions)
    print('=======\n\n', ground_truths[:LIMIT + 1])
    print('=======\n\n', questions[:LIMIT + 1])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a QA model on the CodeRepoQA Python dataset."
    )
    parser.add_argument(
        "--model", type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model name or path for text generation"
    )
    parser.add_argument(
        "--csv", type=str,
        default="../dataset/coderepoqa_python.csv",
        help="Path to the CSV file with 'question' and 'answer' columns"
    )
    args = parser.parse_args()

    print('Loading gen pipeline')
    gen_pipeline = load_generation_pipeline(args.model)
    print('Running eval')
    run_evaluation(args.csv, gen_pipeline)
