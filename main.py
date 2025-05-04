import argparse
from model.eval import load_generation_pipeline, load_opencoder_generation_pipeline, run_evaluation
from model.open_coder import OpenCoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a QA model on the CodeRepoQA Python dataset with GPU batching."
    )
    parser.add_argument("--model", type=str,
                        default="microsoft/phi-2",
                        help="Model name or path for text generation")
    parser.add_argument("--judge_model", type=str,
                        default="Qwen/Qwen1.5-7B-Chat",
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
    # args = parser.parse_args()

    # print('Loading base generation pipeline...')
    # gen_pipeline = load_generation_pipeline(args.model, batch_size=args.batch_size)

    # Tonight: rerank init (no CoT), rerank init (with CoT), rerank refined (no CoT)
    # Tomorrow morning: rerank refined (with CoT), naive RAG (no CoT), naive RAG (with CoT)

    print('Loading judge pipeline...')
    judge_pipeline = load_generation_pipeline(args.judge_model, batch_size=args.batch_size)

    # Reranker Options
    print('loading opencoder generation pipeline (rerank init, no cot)...')
    rerank_init_no_cot = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size, rerank_initial=True, cot=False)

    print('loading opencoder generation pipeline (rerank init, with cot)...')
    rerank_init_with_cot = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size, rerank_initial=True, cot=True)

    print('loading opencoder generation pipeline (rerank refined, no cot)...')
    rerank_refined_no_cot = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size, rerank_refined=True, cot=False)

    
    
    print('\n\nRunning batched evaluation on OpenCoder (rerank init, no cot)...')
    run_evaluation(args.csv, rerank_init_no_cot, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline)

    print('\n\nRunning batched evaluation on OpenCoder (rerank init, with cot)...')
    run_evaluation(args.csv, rerank_init_with_cot, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline)

    print('\n\nRunning batched evaluation on OpenCoder (rerank refined, no cot)...')
    run_evaluation(args.csv, rerank_refined_no_cot, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline)



    # Naive Options - VINCENT MAKE SURE TO RUN THE RIGHT THING
    # print('loading opencoder generation pipeline (naive, no cot)')
    # naive_oc_gen_pipeline = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size, use_naive=True)
    # print('loading opencoder generation pipeline (naive, with cot)')
    # naive_cot_oc_gen_pipeline = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size, cot=True, use_naive=True)

    # print('loading opencoder generation pipeline (no cot)...')
    # oc_gen_pipeline = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size)

    # print('Loading OpenCoder generation pipeline (with CoT)...')
    # cot_oc_gen_pipeline = load_opencoder_generation_pipeline(args.model, batch_size=args.batch_size, cot=True)

    

    # print('\n\nRunning batched evaluation on base...')
    # run_evaluation(args.csv, gen_pipeline, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline)

    # print('\n\nRunning batched evaluation on OpenCoder (no CoT)...')
    # run_evaluation(args.csv, oc_gen_pipeline, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline)

    # print('\n\nRunning batched evaluation on OpenCoder (with CoT)...')
    # run_evaluation(args.csv, cot_oc_gen_pipeline, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline)
