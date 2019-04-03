from scipy import spatial
import math
import numpy as np
from IPython import embed

from utils import *

def calculate_repr_score(i, sentence_vectors, delta):
    sent_num = len(sentence_vectors)
    current_v = sentence_vectors[i]
    score = 0
    for j in range(sent_num):
        if j == i:
            continue
        v = sentence_vectors[j]
        sim = 1 - spatial.distance.cosine(current_v, v)
        score += int(sim > delta)
    score /= sent_num
    return score

def calculate_div_score(i, sentence_vectors, repr_scores):
    higher_index = set()
    sent_num = len(sentence_vectors)
    for j in range(sent_num):
        if j == i:
            continue
        if repr_scores[j] > repr_scores[i]:
            higher_index.add(j)
    if len(higher_index) > 0:
        sim_scores = [1 - spatial.distance.cosine(sentence_vectors[j], sentence_vectors[i]) for j in higher_index]
        score = 1 - max(sim_scores)
    else:
        sim_scores = [1 - spatial.distance.cosine(sentence_vectors[j], sentence_vectors[i]) \
                      for j in range(sent_num) if j != i]
        score = 1 - min(sim_scores)
    return score

def calculate_len_score(i, el, rl):
    score = el[i] / max(el) * math.log(max(rl) / rl[i])
    return score

def select_sentences(budget, scores, sentences):
    sent_num = len(sentences)
    chosen = []
    current_len = 0
    ranked_list = [(i, scores[i]) for i in scores]
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    for i in range(sent_num):
        index = ranked_list[i][0]
        if len(sentences[index]) + current_len > budget:
            continue
        current_len += len(sentences[index])
        chosen.append(index)
    return chosen

def main():
    delta = 0.22
    budget = 665
    para = 0.5
    beta = 0.1
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    #corpusIn = open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.txt')
    #target_set = [5804, 5803, 17361, 20859, 18942, 18336, 21233, 19615, 17945]
    #passages, raw_sentences = load_corpus(corpusIn, target_set, stopword_path)
    ret = {}
    for s in duc_set:
        #if os.path.exists('res/' + s + '.txt'):
        #    continue
        passages, raw_sentences = generate_duc_docs(s, stopword_path)
        similarity_scores = calculate_similarity(passages)

        sent_num = len(passages)
        el, rl = calculate_len(passages, raw_sentences)
        sentence_vecs = construct_sentence_vec(passages)
        repr_scores = {}
        len_scores = {}
        div_scores = {}
        scores = {}
        for i in tqdm(range(sent_num)):
            repr_scores[i] = calculate_repr_score(i, sentence_vecs, delta)
            len_scores[i] = calculate_len_score(i, el, rl)
        for i in tqdm(range(sent_num)):
            div_scores[i] = calculate_div_score(i, sentence_vecs, repr_scores)
            #scores[i] = repr_scores[i]
            scores[i] = math.log(div_scores[i] + 0.01) + math.log(len_scores[i] + 0.01) + math.log(repr_scores[i] + 0.01)

        '''
        chosen = select_sentences(budget, scores, raw_sentences)
        summary = ''
        for i in chosen:
            summary += ' ' + raw_sentences[i]
        '''

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

        l = [-1 for _ in passages]
        for i in summary_id:
            l[i] = 1
        ret[s] = l
    return ret

def calculate_similarity(passages):
    sent_num = len(passages)
    similarity_scores = np.zeros([sent_num, sent_num])
    for i in range(sent_num):
        for j in range(sent_num):
            if j < i:
                similarity_scores[i][j] = similarity_scores[j][i]
            else:
                sim = 0
                for word in passages[i]:
                    if word in passages[j]:
                        sim += 1
                sim /= (math.log(len(set(passages[i])) + 1) + math.log(len(set(passages[j])) + 1))
                #sim = len(set(passages[i]) & set(passages[j])) / (math.log(len(set(passages[i])) + 1) + math.log(len(set(passages[j])) + 1))
                similarity_scores[i][j] = sim
    return similarity_scores

if __name__ == '__main__':
    main()