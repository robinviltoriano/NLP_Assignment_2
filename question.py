import pandas as pd
from utils import clean_text
import faiss

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)

index = faiss.read_index('data_article.index')
data_chunk = pd.read_csv('data_chunk.csv')

def fetch_data_info(dataframe_idx, score):

    '''Data should be data_chunk'''
    info = data_chunk.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['id'] = info['id']
    meta_dict['article'] = info['article']
    meta_dict['score'] = score

    return meta_dict
  
def search(query, top_k, index, model):

    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)

    top_k_ids = list(top_k[1].tolist()[0])
    score = list(top_k[0].tolist()[0])

    results =  [fetch_data_info(idx, score) for idx, score in zip(top_k_ids, score)]

    return results

def cross_score(model_inputs):
    scores = cross_model.predict(model_inputs)
    return scores

def query_answer(query, top_k=10):
    # query = "Who is the vice chairman of Samsung?"
    query = clean_text(query)

    # Search top 20 related documents
    results = search(query, top_k=20, index=index, model=model)

    # Sort the scores in decreasing order
    model_inputs = [[query, result['article']] for result in results]
    scores = cross_score(model_inputs)
    ranked_results = [{'id': result['id'], 'article': result['article'], 'score': score} for result, score in zip(results, scores)]
    ranked_results = sorted(ranked_results, key=lambda x: x['score'], reverse=True)
    
    return pd.DataFrame(ranked_results[:top_k])