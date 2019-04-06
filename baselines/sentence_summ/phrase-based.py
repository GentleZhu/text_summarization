from utils import *
from IPython import embed
import pickle
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy import spatial
import sys

def mmr(similarity_scores, phrase_scores, budget, passages, raw_sentences):
    para = 0.5
    beta = 0.4
    sent_num = len(raw_sentences)

    scores = {}
    for idx, sentence in enumerate(passages):
        score = sum([phrase_scores[phrase] if phrase in phrase_scores else 0 for phrase in sentence ])
        scores[idx] = score

    chosen = [False for _ in raw_sentences]
    summary_id = []
    current_len = 0
    while current_len < budget:
        maxscore = -9999
        pick = -1
        for i in range(sent_num):
            if chosen[i]:
                continue
            tmp_score = scores[i]
            for j in summary_id:
                if scores[i] - similarity_scores[i][j] * scores[j] * para < tmp_score:
                    tmp_score = scores[i] - similarity_scores[i][j] * scores[j] * para
            new_score = tmp_score / math.pow(len(passages[i]), beta)
            if new_score > maxscore and len(raw_sentences[i]) + current_len < budget:
                maxscore = new_score
                pick = i
        if pick == -1:
            break
        current_len += len(raw_sentences[pick])
        chosen[pick] = True
        summary_id.append(pick)
    return summary_id

def select_sentences(phrase_scores, budget, passages, raw_sentences):
    lambda_ = 100
    c = 0.1
    sent_scores = []
    sent_phrases = defaultdict(set)
    for idx, sentence in enumerate(passages):
        sent_phrases[idx] = set(sentence)
        score = sum([phrase_scores[phrase] if phrase in phrase_scores else 0 for phrase in sentence ])
        sent_scores.append(score)
    chosen = list()
    current_len = 0
    current_p = set()
    while len(chosen) < budget:
        max_ = -1
        max_idx = -1
        for idx in range(len(passages)):
            if idx in chosen:
                continue
            addtional_cnt = 0
            for p in sent_phrases[idx]:
                if p in phrase_scores and p not in current_p:
                    addtional_cnt += 1
            score_gain = sent_scores[idx] + lambda_ * (len(current_p) + addtional_cnt) / len(phrase_scores)
            #score_gain /= math.pow(len(raw_sentences[idx]), c)
            #score_gain = sent_scores[idx]
            if score_gain > max_:
                max_ = score_gain
                max_idx = idx
        if max_idx != -1:
            chosen.append(max_idx)
            #print(max_idx, raw_sentences[max_idx])
            current_len += len(raw_sentences[max_idx])
            for p in sent_phrases[max_idx]:
                if p in phrase_scores:
                    current_p.add(p)
            #print(sent_phrases[max_idx])
            #print('current #kw is {}'.format(current_p))
        else:
            break
    return chosen

def calculate_similarity(passages, raw_sentences):
    sent_num = len(passages)
    #vectorizer = TfidfVectorizer()
    #sent_X = vectorizer.fit_transform(raw_sentences).todense()
    similarity_scores = np.zeros([sent_num, sent_num])
    for i in range(sent_num):
        for j in range(sent_num):
            if j < i:
                similarity_scores[i][j] = similarity_scores[j][i]
            else:
                #sim = 0
                #for word in passages[i]:
                #    if word in passages[j]:
                #        sim += 1
                #sim /= (math.log(len(set(passages[i])) + 1) + math.log(len(set(passages[j])) + 1))
                sim = len(set(passages[i]) & set(passages[j])) / (math.log(len(set(passages[i])) + 1) + math.log(len(set(passages[j])) + 1))
                similarity_scores[i][j] = sim
                #similarity_scores[i][j] = 1 - spatial.distance.cosine(sent_X[i], sent_X[j])
    return similarity_scores

def generate_results(prefix):
    #duc_set =  ['d30001', 'd30002', 'd30003', 'd30005','d30006', 'd30007', 'd30015', 'd30033', 'd30034', 'd30036']
    duc_set =  ['2018_ca_wildfire', 'Indiahomo_201809', 'Mars_201807', 'Roaster_201802','Tsunami_201812', 'Bridge_201810', 'James_201808', 'Rhino_201803', 'Tigerwoods_201809', 'final_debate']
    ap = False
    if ap:
        budget = 10000
    else:
        budget = 3
    ret = {}
    for idx, s in enumerate(duc_set):
        phrase_scores = pickle.load(open('/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/data/phrase_scores_{}_{}.p'.format(prefix, s), 'rb'))
        print(idx, s)
        #phrase_scores = pickle.load(open('results/' + s + '.p', 'rb'))
        #file_path = '/shared/data/qiz3/text_summ/data/DUC04/train/' + s + '.txt'
        file_path ='/shared/data/qiz3/text_summ/text_summarization/data/news_doc_line/' + s + '.txt'
        passages, raw_sentences = generate_docs_autophrase(file_path)
        if True:
            chosen = select_sentences(phrase_scores, budget, passages, raw_sentences)
        else:
            similarity_scores = calculate_similarity(passages, raw_sentences)
            chosen = mmr(similarity_scores, phrase_scores, budget, passages, raw_sentences)
        l = [-1 for _ in passages]

        if ap:
            for idx, i in enumerate(chosen):
                l[i] = len(chosen) - idx
            ret[s] = l
        else:
            for i in chosen:
                l[i] = 1
            ret[s] = l

    pickle.dump(ret, open('/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/data/{}_phrase_res.p'.format(prefix), 'wb'))

def main_1():
    budget = 665
    phrase_scores = pickle.load(open('phrase_scores.p', 'rb'))
    for idx, s in enumerate(duc_set):
        print('Running algorithm for doc %d...' % idx)
        passages, raw_sentences = generate_duc_docs_autophrase(s)
        phrase_scores_s = phrase_scores[s]
        similarity_scores = calculate_similarity(passages, raw_sentences)
        #chosen = select_sentences(phrase_scores_s, budget, passages, raw_sentences)
        chosen = mmr(similarity_scores, phrase_scores_s, budget, passages, raw_sentences)
        summary = ''
        for i in chosen:
            summary += raw_sentences[i]
        f = open('tmp/system/' + s + '.txt', 'w')
        f.write(summary)
        f.close()

def main():
    budget = 665
    phrase_scores = pickle.load(open('phrase_scores.p', 'rb'))
    file_path = '/shared/data/qiz3/text_summ/text_summarization/results/2018_ca_wildfire.txt'
    passages, raw_sentences = generate_docs_autophrase(file_path)
    #similarity_scores = calculate_similarity(passages, raw_sentences)
    chosen = select_sentences(phrase_scores, budget, passages, raw_sentences)
    #chosen = mmr(similarity_scores, phrase_scores, budget, passages, raw_sentences)
    summary = ''
    for i in chosen:
        summary += raw_sentences[i]
    exit()

if __name__ == '__main__':
    generate_results(sys.argv[1])
    exit()

    budget = 2000
    phrase_scores = pickle.load(open(sys.argv[1], 'rb'))
    #passages = pickle.load(open(sys.argv[2], 'rb'))
    file_path = '/shared/data/qiz3/text_summ/text_summarization/results/2018_ca_wildfire.txt'
    passages, raw_sentences = generate_docs_autophrase(file_path)
    #raw_sentences = [' '.join(p) for p in passages]
    #similarity_scores = calculate_similarity(passages, raw_sentences)
    if True:
        chosen = select_sentences(phrase_scores, budget, passages, raw_sentences)
        #chosen = mmr(similarity_scores, phrase_scores, budget, passages, raw_sentences)
        summary = ''
        for idx,i in enumerate(chosen):
            summary += raw_sentences[i]
            print(idx, raw_sentences[i])
    else:
        similarity_scores = calculate_similarity(passages, raw_sentences)
        #chosen = select_sentences(phrase_scores_s, budget, passages, raw_sentences)
        chosen = mmr(similarity_scores, phrase_scores, budget, passages, raw_sentences)
        summary = ''
        for i in chosen:
            print(raw_sentences[i])
            #summary += raw_sentences[i]
    embed()
    exit()