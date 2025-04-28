# 2. Import libraries
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein

# 3. Define Evaluation Function
def evaluate_predictions(predictions, ground_truths):
    """
    predictions: list of generated answers
    ground_truths: list of reference answers
    """
    assert len(predictions) == len(ground_truths), "Mismatch in number of predictions and references!"

    smooth_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    edit_similarities = []

    for pred, gt in zip(predictions, ground_truths):
        # BLEU score (simple whitespace split)
        pred_tokens = pred.lower().split()
        gt_tokens   = gt.lower().split()
        bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        # ROUGE scores
        rouge_scores = rouge.score(pred, gt)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        # Edit similarity
        edit_similarity = Levenshtein.ratio(pred, gt)
        edit_similarities.append(edit_similarity)

    # Aggregate results
    results = {
        'BLEU':            sum(bleu_scores)       / len(bleu_scores),
        'ROUGE-1':         sum(rouge1_scores)     / len(rouge1_scores),
        'ROUGE-L':         sum(rougeL_scores)     / len(rougeL_scores),
        'Edit Similarity': sum(edit_similarities) / len(edit_similarities)
    }
    return results
