from utils import *
from IPython import embed
import pickle

def select_sentences(phrase_scores, budget, passages, raw_sentences):
    lambda_ = 60
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
    passages, raw_sentences = generate_duc_docs_autophrase('d30048t')
    phrase_scores = pickle.load(open('phrase_score.p', 'rb'))
    embed()
    exit()

if __name__ == '__main__':
    main()