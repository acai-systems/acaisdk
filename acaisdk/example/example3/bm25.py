import numpy as np
from tqdm import tqdm
from gensim.summarization.bm25 import BM25

def get_bm25_scores(dataset):
    bm25_scores = list()
    labels = list()

    for article in tqdm(dataset['data']):
        for qas_context in article['paragraphs']:
            for qas in qas_context['qas']:
                question_sentence = qas['question_lemmas_without_stopwords']
                candidate_sentences = qas_context['context_sents_lemmas_without_stopwords']
                labels.append(set([d['ans_label'] for d in qas['answers']]))

                bm25 = BM25(candidate_sentences)
                candidate_scores = bm25.get_scores(question_sentence)
                candidate_scores_dict = dict(zip(range(len(candidate_scores)),
                                                 candidate_scores))

                sorted_scores_id = sorted(candidate_scores_dict,
                                          key=candidate_scores_dict.get,
                                          reverse=True)

                bm25_scores.append(sorted_scores_id)
    return (bm25_scores, labels)


# output to outfile
def get_bm25_mapback(dataset, preds, y, k=1):
    res = []
    i = 0
    for article in tqdm(dataset['data']):
        for qas_context in article['paragraphs']:
            currContext = qas_context['context']
            currContextSent = qas_context['context_sent']
            for qas in qas_context['qas']:
                id = qas['id']
                currRes = {}
                currRes['id'] = id
                currRes['question'] = qas['question']
                currRes['context'] = currContext
                currRes['bm25AnswerSent'] = currContextSent[preds[i][0]]
                currRes['goalSents'] = []
                currRes['answersText'] = []
                currRes['goalSents'].append([currContextSent[d['ans_label']] for d in qas['answers']])
                currRes['answersText'].append([d['text'] for d in qas['answers']])
                currRes['goalSents'] = currRes['goalSents'][0]
                currRes['answersText'] = currRes['answersText'][0]
                res.append(currRes)
                i += 1
    return res





