# OpenCoder: Synthesizing Stack Overflow Data with Retrieval-Augmented LLMs

---

## Overview

OpenCoder, is a lightweight RAG pipeline designed to enhance LLM performance on software engineering question-answering tasks. It combines 1) semantic or keyword retrieval from the public Stack Overflow dump, 2) a self-refinement feedback loop, 3) chain-of-thought prompting, and 4) an output-reranking stage. It is built on the 1.5B–parameter PHI- 2 model.

---

## Notice

All development for this project was done on the Yale Mccleary Cluster with the following settings
* Number of CPU cores per node: 1
* Memory per CPU core in GiB: 45
* Partitions: gpu_devel
* Number of GPUs per node: 2
* GPU: NVIDIA GeForce RTX 3090

---

## Dependencies

For convenience, we have created a `env.yml` file that can be run to install all but one required dependency with the following command, creating the opencoder_env environment: 

```bash
conda env create -f env.yml
```

Then, activate the environment with:

```bash
conda activate opencoder_env
```

Finally, run the following command for GPU acceleration:

```bash
pip install accelerate
```

At a glance, this will create an environment with the following:
* python=3.10
* pytorch-cuda
* accelerate
* faiss-cpu
* sentence-transformers
* nltk
* pandas
* transformers
* rouge-score
* python-levenshtein

---

## Pretrained Models & Datasets

* **Generation Models**:

  * `microsoft/phi-2` (default base generation)
  * `Qwen/Qwen1.5-7B-Chat` (judge model)
* **Sentence Transformer**:

  * `all-MiniLM-L6-v2` for semantic retrieval embeddings
* **Dataset**:

  * `dataset/coderepoqa_python.csv` (CodeRepoQA Python question-answer pairs)
    * Compiled from https://github.com/kinesiatricssxilm14/CodeRepoQA where all Python relevant datasets were combined into one CSV
  * `dataset/question_answer.csv` (for generic RAG)
    * Compiled from https://www.kaggle.com/datasets/stackoverflow/stackoverflow via BigQuery using the following SQL query:
        ```sql
        SELECT
        q.id AS question_id,
        q.title AS question_title,
        q.body AS question_body,
        q.tags AS question_tags,
        a.id AS answer_id,
        a.body AS answer_body,
        a.score AS answer_score
        FROM
        `bigquery-public-data.stackoverflow.posts_questions` AS q
        JOIN
        `bigquery-public-data.stackoverflow.posts_answers` AS a
        ON q.id = a.parent_id
        WHERE
        q.creation_date >= '2020-01-01'
        AND q.score >= 5
        AND a.score >= 5
        AND (q.tags LIKE '%python%' OR q.tags LIKE '%javascript%' OR q.tags LIKE '%java%')
        LIMIT 20000
        ```

All models are automatically downloaded from the Hugging Face Hub within our code. The datasets are placed under `dataset/`

---

## How to Run

1. **Run evaluation script:**

   ```bash
   python3 main.py --limit 800 --opencoder
   ```

   Further Customizable Flags:
   * `--limit [LIMIT]`: number of questions to evaluate model on
   * `--batch_size [BATCH SIZE]`: number of questions processed in parallel (GPU only)
   * `--opencoder`: use opencoder framework (must include to run with the customizable flags below). If absent, uses baseline model.
   * `--rag [semantic|keyword]`: enhance model with either semantic or keyword RAG. Default semantic.
   * `--cot`: enhance model with CoT
   * `--rerank [initial|refined]`: enhance model with output reranking, either in the draft stage or after the feedeback stage. Default None.

    Examples:

      * Keyword RAG:
        ```bash
        python3 main.py --limit 800 --opencoder --rag keyword
        ```
      * Use CoT:
        ```bash
        python3 main.py --limit 800 --opencoder --cot
        ```
      * Use CoT and reranking refined
        ```bash
        python3 main.py --limit 800 --opencoder --cot --rerank refined
        ```

2. **Output:**
   * The script prints inference progress and final metrics (including LLM-judge score, BLEU, ROUGE-1, ROUGE-L, and edit distance scores).

