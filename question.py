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

def query_answer(query, top_k=5):
    # query = "Who is the vice chairman of Samsung?"
    query = clean_text(query)

    results = search(query, top_k=10, index=index, model=model)

    model_inputs = [[query, item['article']] for item in results]
    scores = cross_score(model_inputs)

    #Sort the scores in decreasing order
    ranked_results = [{'Id': inp['id'], 'Score': score} for inp, score in zip(results, scores)]
    ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)
    
    # get the top k article ids
    article_id = []
    for id in ranked_results[:top_k]:
        article_id.append(id['Id'])
    
    # combine the output into a single string
    output = ''
    for art_chunk in results:
        if art_chunk['id'] in article_id:
            output += f"From article id {str(art_chunk['id'])}, {art_chunk['article']}. "
            
    return output