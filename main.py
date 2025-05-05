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
                        default="./dataset/coderepoqa_python.csv",
                        help="Path to the CSV file with 'question' and 'answer' columns")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="Number of questions to process in parallel on GPU")
    parser.add_argument("--limit", type=int,
                        default=None,
                        help="Optional limit on number of samples for quick testing")
    parser.add_argument("--opencoder", action='store_true',
                        help="use opencoder framework. if not present, use baseline")
    parser.add_argument("--rag", choices=['semantic', 'keyword'],
                        default="semantic",
                        help="keyword or semantic rag. default semantic")
    parser.add_argument("--cot", action='store_true',
                        help="if present, using cot prompting")
    parser.add_argument("--rerank", choices=['initial', 'refined'],
                        default=None,
                        help="Choose 'initial' for rerank after initial generation or 'refined' for rerank after refined response. default none")
    args = parser.parse_args()

    print('Loading base generation pipeline...')
    base_pipeline = load_generation_pipeline(args.model, batch_size=args.batch_size)

    print('Loading judge pipeline...')
    judge_pipeline = load_generation_pipeline(args.judge_model, batch_size=args.batch_size)

    eval_pipeline = base_pipeline
    model_name = "OpenCoder (" if args.opencoder else "baseline..."
    if args.opencoder:
        print('loading opencoder generation pipeline...')
        eval_pipeline = load_opencoder_generation_pipeline(base_pipeline, limit=args.limit)

        r_init_flag = args.rerank == 'initial'
        r_ref_flag = args.rerank == 'refined'
        use_naive_rag = args.rag == 'keyword'
        
        model_name += ("with cot" if args.cot else "no cot")
        model_name += (", keyword rag" if use_naive_rag else ", semantic rag")
        if r_init_flag:
            model_name += ", rerank initial"
        if r_ref_flag:
            model_name += ", rerank refined"
        model_name += ")..."

    print(f'\nRunning batched evaluation on {model_name}')
    if args.opencoder:
        run_evaluation(args.csv, eval_pipeline, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline, rerank_initial=r_init_flag, rerank_refined=r_ref_flag, use_cot=args.cot, use_naive=use_naive_rag)
    else:
        run_evaluation(args.csv, eval_pipeline, batch_size=args.batch_size, limit=args.limit, judge_pipeline=judge_pipeline, baseline=True)

