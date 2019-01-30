from utils import *
from IPython import embed
import pickle
from collections import defaultdict
import math

def select_sentences(phrase_scores, budget, passages, raw_sentences):
    lambda_ = 4
    c = 0.5
    sent_scores = {}
    sent_phrases = defaultdict(set)
    for idx, sentence in enumerate(passages):
        sent_phrases[idx] = set(sentence)
        score = sum([phrase_scores[phrase] for phrase in set(sentence)])
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

def main():
    budget = 665
    phrase_scores = pickle.load(open('phrase_scores.p', 'rb'))
    for idx, s in enumerate(duc_set):
        print('Running algorithm for doc %d...' % idx)
        passages, raw_sentences = generate_duc_docs_autophrase(s)
        phrase_scores_s = phrase_scores[s]
        chosen = select_sentences(phrase_scores_s, budget, passages, raw_sentences)
        summary = ''
        for i in chosen:
            summary += raw_sentences[i]
        f = open('tmp/system/' + s + '.txt', 'w')
        f.write(summary)
        f.close()

if __name__ == '__main__':
    main()