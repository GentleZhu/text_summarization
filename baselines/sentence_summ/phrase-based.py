from utils import *
from IPython import embed
import pickle
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy import spatial

def mmr(similarity_scores, phrase_scores, budget, passages, raw_sentences):
    para = 0.5
    beta = 0.4
    sent_num = len(raw_sentences)

    scores = {}
    for idx, sentence in enumerate(passages):
        score = sum([phrase_scores[phrase] for phrase in sentence])
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
    lambda_ = 4
    c = 0.5
    sent_scores = {}
    sent_phrases = defaultdict(set)
    for idx, sentence in enumerate(passages):
        sent_phrases[idx] = set(sentence)
        score = sum([phrase_scores[phrase] for phrase in sentence])
        sent_scores[idx] = score

    chosen = list()
    current_len = 0
    current_p = set()
    while current_len < budget:
        max_ = -1
        max_idx = -1
        for idx in range(len(passages)):
            if idx in chosen or current_len + len(raw_sentences[idx]) > budget:
                continue
            score_gain = sent_scores[idx] + lambda_ * len(sent_phrases[idx] - current_p)
            score_gain /= math.pow(len(raw_sentences[idx]), c)
            if score_gain > max_:
                max_ = score_gain
                max_idx = idx
        if max_idx != -1:
            chosen.append(max_idx)
            current_len += len(raw_sentences[max_idx])
            current_p |= sent_phrases[max_idx]
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

def main():
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

if __name__ == '__main__':
    main()