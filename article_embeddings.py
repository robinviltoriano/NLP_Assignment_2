import pandas as pd
import numpy as np
from utils import clean_text
from tqdm import tqdm
import faiss

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

def get_data(csv_file):

    df = pd.read_csv(csv_file, encoding='latin-1')
    df = df[['id', 'article']]

    df = df[df.duplicated(subset=['article'], keep=False)]
    df = df.drop_duplicates(subset=['article'],keep='first').reset_index(drop=True)

    return df

def chunk_text(data_index, data_text, chunk_size, chunk_overlap):

    list_chunk_text = []

    for position in range(len(data_index)):

        words = clean_text(data_text[position]).split()

        start = 0
        part = 1
        while start < len(words):
            end = start + chunk_size
            segment = ' '.join(words[start:end])
            list_chunk_text.append((str(data_index[position]) + str(part), segment))
            part += 1
            start += (chunk_size - chunk_overlap)
            
    data_chunk = pd.DataFrame(list_chunk_text, columns=['id', 'article'])

    data_chunk.to_csv('data_chunk.csv', index=False)

    return data_chunk

def data_prep(csv_file):
    data = get_data(csv_file)
    data['article'] = data['article'].apply(clean_text)
    data_chunk = chunk_text(data['id'], data['article'], 500, 50)
  
    encoded_data = model.encode(data_chunk['article'].tolist())
    encoded_data = np.asarray(encoded_data.astype('float32'))

    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(encoded_data, np.array(range(0, len(data_chunk))))
    faiss.write_index(index, 'data_article.index')



def main():
    data_prep('./news_dataset.csv')

    

