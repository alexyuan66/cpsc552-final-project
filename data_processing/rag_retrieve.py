# 2. Import libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def retrieve_semantic(df, query, tag_keywords, model, index, top_k=5, boost_factor=5):
    """
    Retrieve relevant question-answer pairs given a query
    """

    results = []

    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # Search similar questions
    D, I = index.search(query_embedding, top_k * 2)
    
    # Iterate through all similar questions
    for idx, score in zip(I[0], D[0]):
        # Skip if no match
        if idx == -1:
            continue
            
        item = {
            'question_title': df.iloc[idx]['question_title'],
            'question_body': df.iloc[idx]['question_body'],
            'answer_body': df.iloc[idx]['answer_body'],
            'tags': df.iloc[idx]['question_tags'],
            'semantic_score': score
        }
        
        # Boost score if the question's tags match the desired keywords
        tags = str(df.iloc[idx]['question_tags']).lower()
        for keyword in tag_keywords:
            if keyword in tags:
                item['semantic_score'] += boost_factor
                break  # Boost once per keyword match
        
        results.append(item)
    
    # Get results
    results = sorted(results, key=lambda x: x['semantic_score'], reverse=True)
    return results[:top_k]

def extract_keywords_from_query(query):
    """
    Extract important keywords from a users query
    """

    tokens = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    keywords = [token for token in tokens if token.isalnum() and token not in stop_words]
    return keywords

def main(dataset, query, tag_keywords):
    # Preprocess everything
    df = pd.read_csv(dataset)
    df['text'] = df['question_title'].fillna('') + " " + df['question_body'].fillna('') # concatenates question and body
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode questions
    print("Encoding all questions...")
    corpus_embeddings = model.encode(df['text'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    # FAISS index
    d = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(d) # cosine similarity
    index.add(corpus_embeddings)
    print(f"Finished indexing {len(corpus_embeddings)} question-answer pairs")

    # Get results and display
    results = retrieve_semantic(df, query, tag_keywords, model, index, top_k=5)
    for r in results:
        print("="*100)
        print("QUESTION:", r['question_title'])
        print("TAGS:", r['tags'])
        print("ANSWER:", r['answer_body'])
        print("SCORE:", r['semantic_score'])



if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    dataset = './dataset/question_answer.csv'
    query = 'How do I fix an index out of range error in Python?'
    tag_keywords = extract_keywords_from_query(query)
    main(dataset, query, tag_keywords)