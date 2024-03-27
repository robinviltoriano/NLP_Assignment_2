from question import query_answer
from question import search
import pandas as pd

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

import faiss
index = faiss.read_index('data_article.index')

question_data = pd.read_csv('question_test_data.csv')
question_data

def mrr_score(answers, queries):
    '''answers is a list of list of ids'''
    score = []
    for i, answer in enumerate(answers):
        for j, index in enumerate(answer):
            if index == queries[i]:
                score.append(1 / (j + 1))
                break
    return sum(score) / len(score) if len(score) > 0 else 0

def accuracy_score(answers, queries):
    '''answers is a list of list of ids'''
    score = []
    for i, answer in enumerate(answers):
        for j, index in enumerate(answer):
            if index == queries[i]:
                score.append(1)
                break
    return sum(score) / len(score) if len(score) > 0 else 0

def eval(question_data, index, model):
    # Accuracy
    answer_ids = []
    for i, question in enumerate(question_data['question']):
        answer = search(question_data['question'][i], top_k=20, index=index, model=model)
        answers = [answer[x]['id'] // 10 for x in range(len(answer))]
        answer_ids.append(answers)
    accuracy_score = accuracy_score(answer_ids, question_data['doc_id'])

    # MRR search
    mrr_search = mrr_score(answer_ids, question_data['doc_id'])

    # MRR rerank
    rerank_data = pd.read_csv('answers_reranked.csv')
    question_article_ids = {}
    for i, question_id in enumerate(rerank_data['question_id']):
        if question_id not in question_article_ids:
            question_article_ids[question_id] = [rerank_data['article_id'][i]]
        else:
            question_article_ids[question_id].append(rerank_data['article_id'][i])

    rerank_article_ids = [question_article_ids[x] for x in question_article_ids]
    mrr_rerank = mrr_score(rerank_article_ids, question_data['doc_id'])

    print('Accuracy search:', accuracy_score)
    print('MRR search:', mrr_search)
    print('MRR rerank:', mrr_rerank)
    