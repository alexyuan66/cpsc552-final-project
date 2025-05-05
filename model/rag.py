# 2. Import libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np




class RAG:
    def __init__(self, dataset='./dataset/question_answer.csv', limit=None):
        self.dataset = dataset
        if limit is not None:
            self.df = pd.read_csv(dataset, nrows=limit)
        else:
            self.df = pd.read_csv(dataset)

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Preprocess everything
        self.df['text'] = self.df['question_title'].fillna('') + " " + self.df['question_body'].fillna('') # concatenates question and body
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

        # Encode questions
        print("Encoding all questions...")
        corpus_embeddings = self.model.encode(self.df['text'].tolist(), show_progress_bar=True, normalize_embeddings=True)

        # FAISS index for sentence transformer retrieval
        d = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # cosine similarity
        self.index.add(corpus_embeddings)
        print(f"Finished indexing {len(corpus_embeddings)} question-answer pairs")

        # TF-IDF matrix over dataset for naive retrieval
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'].tolist())

    def _retrieve_semantic_from_embedding(
        self,
        q_emb: np.ndarray,
        tag_keywords,
        top_k=5,
        boost_factor=5,
    ):
        """Same logic as before but receives an *embedding*, not raw text."""
        D, I = self.index.search(q_emb.reshape(1, -1), top_k * 2)

        scored = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            item = {
                "question_title": self.df.iloc[idx]["question_title"],
                "question_body": self.df.iloc[idx]["question_body"],
                "answer_body": self.df.iloc[idx]["answer_body"],
                "tags": self.df.iloc[idx]["question_tags"],
                "semantic_score": float(score),
            }
            tags = str(item["tags"]).lower()
            for kw in tag_keywords:
                if kw in tags:
                    item["semantic_score"] += boost_factor
                    break
            scored.append(item)

        scored.sort(key=lambda x: x["semantic_score"], reverse=True)
        return scored[:top_k]
    
    def __retrieve_semantic_batch(self, query, tag_keywords, top_k=5, boost_factor=5):
        """
        Util function that retrieves relevant question-answer pairs given a query
        """

        results = []

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search similar questions
        D, I = self.index.search(query_embedding, top_k * 2)
        
        # Iterate through all similar questions
        for idx, score in zip(I[0], D[0]):
            # Skip if no match
            if idx == -1:
                continue
                
            item = {
                'question_title': self.df.iloc[idx]['question_title'],
                'question_body': self.df.iloc[idx]['question_body'],
                'answer_body': self.df.iloc[idx]['answer_body'],
                'tags': self.df.iloc[idx]['question_tags'],
                'semantic_score': score
            }
            
            # Boost score if the question's tags match the desired keywords
            tags = str(self.df.iloc[idx]['question_tags']).lower()
            for keyword in tag_keywords:
                if keyword in tags:
                    item['semantic_score'] += boost_factor
                    break  # Boost once per keyword match
            
            results.append(item)
        
        # Get results
        results = sorted(results, key=lambda x: x['semantic_score'], reverse=True)
        return results[:top_k]
    
    def _extract_keywords_batch(self, q: str):
        """Return a list of alphanumeric keywords for a SINGLE query string."""
        tokens = word_tokenize(q.lower())
        stop_words = set(stopwords.words("english"))
        return [t for t in tokens if t.isalnum() and t not in stop_words]
    

    def retrieve(self, query, top_k=5):
        """
        Retrieve top k most relevant question-answer pairs from the given dataset given a query using sentence transformer.

        Args:
            query (str): The query to process.
            dataset (str): Dataset to find question-answer pairs from.

        Returns:
            str: Concatenated string containing the top k most relevant question-answer pairs.
        """
        
        tag_keywords = self._extract_keywords_batch(query)

        # Get results and concatenate
        results = self.__retrieve_semantic_batch(query, tag_keywords, top_k)

        concatenated_results = []
        for i, r in enumerate(results):
            concatenated_results.append(f'{i})')
            concatenated_results.append(f"Question: {r['question_title']}")
            concatenated_results.append(f"Tags: {r['tags'].replace('|', ', ')}")
            concatenated_results.append(f"Answer: {r['answer_body']}")
            concatenated_results.append('\n')

        return '\n'.join(concatenated_results)

    def retrieve_naive(self, query, top_k=5):
        """
        Retrieve top k most relevant question-answer pairs from the given dataset given a query using naive TF-IDF matrix.

        Args:
            query (str): The query to process.

        Returns:
            str: Concatenated string containing the top k most relevant question-answer pairs.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = scores.argsort()[::-1][:top_k]

        concatenated_results = []
        for i, idx in enumerate(top_indices):
            question_title = self.df.iloc[idx]['question_title']
            tags = self.df.iloc[idx]['question_tags']
            answer_body = self.df.iloc[idx]['answer_body']

            concatenated_results.append(f"{i})")
            concatenated_results.append(f"Question: {question_title}")
            concatenated_results.append(f"Tags: {tags.replace('|', ', ')}")
            concatenated_results.append(f"Answer: {answer_body}")
            concatenated_results.append('\n')

        return '\n'.join(concatenated_results)

    def retrieve_batch(self, queries: list[str], top_k=5):
        """
        Retrieve from a *batch* of queries.

        Args:
            queries (list[str]) : list of questions
            top_k (int)         : how many Q‑A pairs per query to return

        Returns:
            list[str] : one formatted string per original query
        """
        if not isinstance(queries, (list, tuple)):
            raise TypeError("`queries` must be a list/tuple of strings.")

        # 1. Compute embeddings on the GPU in one go
        q_embs = self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )

        # 2. Keyword extraction per query (cheap, happens on CPU)
        kw_lists = [self._extract_keywords_batch(q) for q in queries]

        # 3. Search FAISS once for **all** queries
        D, I = self.index.search(q_embs, top_k * 2)

        # 4. Assemble results
        formatted_out = []
        for qi, (row_I, row_D, kw) in enumerate(zip(I, D, kw_lists)):
            scored = []
            for idx, score in zip(row_I, row_D):
                if idx == -1:
                    continue
                item = {
                    "question_title": self.df.iloc[idx]["question_title"],
                    "question_body": self.df.iloc[idx]["question_body"],
                    "answer_body": self.df.iloc[idx]["answer_body"],
                    "tags": self.df.iloc[idx]["question_tags"],
                    "semantic_score": float(score),
                }
                tags = str(item["tags"]).lower()
                for k in kw:
                    if k in tags:
                        item["semantic_score"] += 5  # same boost
                        break
                scored.append(item)

            scored.sort(key=lambda x: x["semantic_score"], reverse=True)
            scored = scored[:top_k]

            # pretty‑print exactly like before
            parts = []
            for j, r in enumerate(scored):
                parts.append(f"{j})")
                parts.append(f"Question: {r['question_title']}")
                parts.append(f"Tags: {r['tags'].replace('|', ', ')}")
                parts.append(f"Answer: {r['answer_body']}")
                parts.append("")  # blank line
            formatted_out.append("\n".join(parts))

        return formatted_out

    def retrieve_batch_naive(self, queries: list[str], top_k=5):
        """
        Retrieve from a batch of queries using TF-IDF (naive method).

        Args:
            queries (list[str]) : list of questions
            top_k (int)         : how many Q‑A pairs per query to return

        Returns:
            list[str] : one formatted string per original query
        """
        if not isinstance(queries, (list, tuple)):
            raise TypeError("`queries` must be a list/tuple of strings.")

        # 1. Transform all queries into TF-IDF vectors
        q_vecs = self.vectorizer.transform(queries)

        # 2. Extract keywords from each query
        kw_lists = [self._extract_keywords_batch(q) for q in queries]

        # 3. Score and format
        formatted_out = []
        for qi, (q_vec, kw) in enumerate(zip(q_vecs, kw_lists)):
            scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
            top_indices = scores.argsort()[::-1][:top_k * 2]

            scored = []
            for idx in top_indices:
                item = {
                    "question_title": self.df.iloc[idx]["question_title"],
                    "question_body": self.df.iloc[idx]["question_body"],
                    "answer_body": self.df.iloc[idx]["answer_body"],
                    "tags": self.df.iloc[idx]["question_tags"],
                    "semantic_score": float(scores[idx]),  # reusing the same key
                }
                tags = str(item["tags"]).lower()
                for k in kw:
                    if k in tags:
                        item["semantic_score"] += 5
                        break
                scored.append(item)

            scored.sort(key=lambda x: x["semantic_score"], reverse=True)
            scored = scored[:top_k]

            # Format
            parts = []
            for j, r in enumerate(scored):
                parts.append(f"{j})")
                parts.append(f"Question: {r['question_title']}")
                parts.append(f"Tags: {r['tags'].replace('|', ', ')}")
                parts.append(f"Answer: {r['answer_body']}")
                parts.append("")  # blank line
            formatted_out.append("\n".join(parts))

        return formatted_out
    

# =====================================================================
# EXAMPLE USAGE:
# =====================================================================
#
# if __name__ == "__main__":
#     query1 = 'How do I fix an index out of range error in Python?'
#     query2 = 'How do I concatenate two strings in Python?'

#     rag = RAG()
#     print(rag.retrieve(query1))
#     print(rag.retrieve(query2))
