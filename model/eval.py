#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

# Ensure data_processing directory is on the Python path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, '../data_processing'))

# Import the evaluation function from the data_processing directory
from coderepoqa_evaluation import evaluate_predictions


def load_generation_pipeline(model_name, batch_size=8, max_new_tokens=256, temperature=0.7):
    """
    Load a text-generation pipeline for the specified model, handling
    authentication, device placement, and padding for batching.
    Returns a Hugging Face pipeline object.
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

    # Ensure tokenizer has a pad token for batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.eos_token_id

    # Create a text-generation pipeline with batching enabled
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return gen_pipeline


def run_evaluation(csv_path, gen_pipeline, batch_size=8, limit=None):
    """
    Read (question, answer) pairs from csv_path, generate model predictions
    using batched inference via the provided generation pipeline, and evaluate
    them via evaluate_predictions.
    """
    questions = []
    ground_truths = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"])
            ground_truths.append(row["answer"])

    if limit is not None:
        questions = questions[:limit]
        ground_truths = ground_truths[:limit]

    predictions = []
    total = len(questions)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_questions = questions[start:end]
        print(f"Processing questions {start + 1} to {end} of {total}")
        batch_results = gen_pipeline(batch_questions)
        for result in batch_results:
            predictions.append(result[0]["generated_text"].strip())

    metrics = evaluate_predictions(predictions, ground_truths)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a QA model on the CodeRepoQA Python dataset with GPU batching."
    )
    parser.add_argument("--model", type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model name or path for text generation")
    parser.add_argument("--csv", type=str,
                        default="../dataset/coderepoqa_python.csv",
                        help="Path to the CSV file with 'question' and 'answer' columns")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="Number of questions to process in parallel on GPU")
    parser.add_argument("--limit", type=int,
                        default=None,
                        help="Optional limit on number of samples for quick testing")
    args = parser.parse_args()

    print('Loading generation pipeline...')
    gen_pipeline = load_generation_pipeline(args.model, batch_size=args.batch_size)
    print('Running batched evaluation...')
    run_evaluation(args.csv, gen_pipeline, batch_size=args.batch_size, limit=args.limit)
