import os
import json
import random
import pandas as pd

# Path to your downloaded and extracted CodeRepoQA Python folder
base_path = './dataset/CodeRepoQA_Python'
if not os.path.exists(base_path):
    print(f"File {base_path} does not exist!")
    exit(1)

questions = []
answers = []

for root, dirs, files in os.walk(base_path):
    if not dirs:
        json_files = [f for f in files if f.endswith('.json')]
        if not json_files:
            continue

        file_count = max(1, len(json_files) // 10)
        sampled_files = random.sample(json_files, file_count)

        for file in sampled_files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    issue = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
            # Build question (title + body)
            title = issue.get('title', '')
            body = issue.get('body', '')
            if not title or not body:
                continue
            question_text = (title + "\n" + body).strip()

            # Find the best answer based on reactions
            comments = issue.get('comments_details', [])
            best_comment = ''
            best_score = -1

            for comment in comments:
                comment_body = comment.get('body', '').strip()
                reactions = comment.get('reactions', {})
                plus_ones = reactions.get('+1', 0) if reactions else 0
                hearts = reactions.get('heart', 0) if reactions else 0
                hoorays = reactions.get('hooray', 0) if reactions else 0
                score = plus_ones + hearts + hoorays

                if comment_body:
                    # Prefer comments with higher number of +1s
                    if score > best_score:
                        best_comment = comment_body
                        best_score = score

            # Only add if both question and answer exist
            if question_text and best_comment:
                questions.append(question_text)
                answers.append(best_comment)

print(f"Total question-answer pairs collected: {len(questions)}")

# Build DataFrame
df_coderepoqa_python = pd.DataFrame({
    'question': questions,
    'answer': answers
})

# Save to CSV
dataset_dir = "./dataset"
output_csv_path = os.path.join(dataset_dir, 'coderepoqa_python.csv')
os.makedirs(dataset_dir, exist_ok=True)
df_coderepoqa_python.to_csv(output_csv_path, index=False, quoting=1, escapechar='\\')
print(f"Saved processed data to {output_csv_path}")
