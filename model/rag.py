# 2. Import libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class RAG:
    def __init__(self, dataset='../dataset/question_answer.csv'):
        self.dataset = dataset
        self.df = pd.read_csv(dataset)

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Preprocess everything
        self.df['text'] = self.df['question_title'].fillna('') + " " + self.df['question_body'].fillna('') # concatenates question and body
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode questions
        print("Encoding all questions...")
        corpus_embeddings = self.model.encode(self.df['text'].tolist(), show_progress_bar=True, normalize_embeddings=True)

        # FAISS index
        d = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # cosine similarity
        self.index.add(corpus_embeddings)
        print(f"Finished indexing {len(corpus_embeddings)} question-answer pairs")

    
    def __retrieve_semantic(self, query, tag_keywords, top_k=5, boost_factor=5):
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
    

    def __extract_keywords_from_query(self, query):
        """
        Util function that extracts important keywords from a user's query
        """

        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [token for token in tokens if token.isalnum() and token not in stop_words]
        return keywords
    

    def retrieve(self, query, top_k=5):
        """
        Retrieve top k most relevant question-answer pairs from the given dataset given a query.

        Args:
            query (str): The query to process.
            dataset (str): Dataset to find question-answer pairs from.

        Returns:
            str: Concatenated string containing the top k most relevant question-answer pairs.
        """
        
        tag_keywords = self.__extract_keywords_from_query(query)

        # Get results and concatenate
        results = self.__retrieve_semantic(query, tag_keywords, top_k)

        concatenated_results = []
        for i, r in enumerate(results):
            concatenated_results.append(f'{i})')
            concatenated_results.append(f"Question: {r['question_title']}")
            concatenated_results.append(f"Tags: {r['tags'].replace('|', ', ')}")
            concatenated_results.append(f"Answer: {r['answer_body']}")
            concatenated_results.append('\n')

        return '\n'.join(concatenated_results)
    

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