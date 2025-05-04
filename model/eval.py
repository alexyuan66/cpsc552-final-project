#!/usr/bin/env python3
import os
import sys
import csv
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import math


from model.open_coder import OpenCoder

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

# Ensure data_processing directory is on the Python path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, '../data_processing'))

# Import the evaluation function from the data_processing directory
from coderepoqa_evaluation import evaluate_predictions

# custom pipline class that is callable in the same way as transformers.pipeline
# purpose: for truncating inputs to the max input length allowed by model, and for more control
class CustomPipeline:
    def __init__(self, model, tokenizer, max_input_tokens=2048, max_new_tokens=256, temperature=0.7, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, prompts):
        # Accept either "one prompt" or ["prompt1", …]
        if isinstance(prompts, str):
            prompts = [prompts]

        all_outputs = []

        for i in range(0, len(prompts), self.batch_size):
            batch_questions = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True, 
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for text in decoded:
                all_outputs.append([{"generated_text": text}])

        return all_outputs

def load_generation_pipeline(model_name, batch_size=8, max_input_tokens=2048, max_new_tokens=256, temperature=0.7):
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
        padding_side='left',

    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # model = AutoModelForCausalLM.from_pretrained(
    # model_name,
    # attn_implementation="flash_attention_2",   # ≈1.8 × speed‑up if supported
    # torch_dtype=torch.float16,
    # device_map="auto",
    # )
    torch.backends.cuda.matmul.allow_tf32 = True    # small boost on Ampere+
    model = torch.compile(model)                   # PyTorch 2.x, nvFuser
    #model.to("cpu")

    # Ensure tokenizer has a pad token for batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.eos_token_id

    # Create a text-generation pipeline with batching enabled
   # gen_pipeline = pipeline(
   #     "text-generation",
   #     model=model,
   #     tokenizer=tokenizer,
   #     device_map="auto" if torch.cuda.is_available() else None,
   #     batch_size=batch_size,
   #     max_new_tokens=max_new_tokens,
   #     do_sample=True,
   #     temperature=temperature,
   # )
    gen_pipeline = CustomPipeline(model, tokenizer, max_input_tokens, max_new_tokens, temperature, batch_size)
    return gen_pipeline


import re

DEFAULT_JUDGE_PROMPT = (
    "Output a score to rate the two answers from 0 (least similar) to 1 "
    "(most similar). Format your answer with LaTeX, like this: \\boxed{score}."
)

def LLM_judge(judge_pipeline, predicted: str, ground_truth: str,
              judge_prompt: str = DEFAULT_JUDGE_PROMPT) -> float:
    """
    Ask an LLM‐judge to score similarity between *predicted* and *ground_truth*.

    Args
    ----
    judge_pipeline : a text‑generation pipeline or CustomPipeline
    predicted      : model output string
    ground_truth   : reference answer string
    judge_prompt   : system prompt that explains the boxed‑score format

    Returns
    -------
    float          : extracted score in [0, 1]  (NaN if not found)
    """
    full_prompt = (
        f"{judge_prompt}\n\n"
        f"Answer 1 (model prediction):\n{predicted}\n\n"
        f"Answer 2 (ground‑truth):\n{ground_truth}\n"
    )

    # Pipeline always expects a *batch* in this repo
    raw = judge_pipeline([full_prompt])[0]
    if isinstance(raw, list):          # CustomPipeline returns list[list[dict]]
        raw_text = raw[0]["generated_text"]
    elif isinstance(raw, dict):
        raw_text = raw["generated_text"]
    else:                              # plain string or other
        raw_text = str(raw)

    # Extract \boxed{…}
    # print(raw_text)
    m = re.search(r"\\boxed\{ *([\d.]+) *\}", raw_text)
    return float(m.group(1)) if m else float("nan")



def run_evaluation(csv_path,
                   gen_pipeline,
                   batch_size=8,
                   limit=None,
                   judge_pipeline=None,
                   judge_prompt=DEFAULT_JUDGE_PROMPT, use_cot: bool = False, rerank_initial: bool = False, rerank_refined: bool = False, use_naive: bool = False):
    """
    Generate answers and (optionally) have an LLM judge rate them.
    """

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    questions      = df["question"].fillna("").tolist()
    ground_truths  = df["answer"].fillna("").tolist()

    if limit is not None:
        questions      = questions[:limit]
        ground_truths  = ground_truths[:limit]

    # ---------- generation ----------
    predictions = []
    total = len(questions)
    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = questions[start:end]
        print(f"Processing questions {start+1}–{end} / {total}")

        batch_out = gen_pipeline(batch, use_cot=use_cot, rerank_initial=rerank_inital, rerank_refined=rerank_refined, use_naive=use_naive)
        for result in batch_out:
            # Accept: "answer", [{"generated_text": …}], or {"generated_text": …}
            if isinstance(result, str):
                predictions.append(result.strip())
            elif isinstance(result, list):
                predictions.append(result[0]["generated_text"].strip())
            elif isinstance(result, dict):
                predictions.append(result["generated_text"].strip())
            else:                       # fall‑back – stringify whatever it is
                predictions.append(str(result).strip())

    # ---------- reference metrics ----------
    metrics = evaluate_predictions(predictions, ground_truths)

    # ---------- optional LLM‑judge ----------
    if judge_pipeline is not None:
        judge_scores = [
            LLM_judge(judge_pipeline, pred, gt, judge_prompt)
            for pred, gt in zip(predictions, ground_truths)
        ]

        # keep only finite numbers
        valid = [s for s in judge_scores if not math.isnan(s)]

        if valid:                       # at least one usable score
            metrics["LLM_judge_mean"] = sum(valid) / len(valid)
            metrics["LLM_judge_n"]    = len(valid)   # (optional) how many kept
        else:                           # every score was NaN
            metrics["LLM_judge_mean"] = float("nan")
            metrics["LLM_judge_n"]    = 0

    print(metrics)
    return metrics



def load_opencoder_generation_pipeline(base_pipeline):
    ## DO NOT INITIALIZE ANOTHER PIPELINE, ADDING TOO MUCH BLOAT
    return OpenCoder(base_pipeline)

