from utils import *
import math
import numpy as np
import pickle
from IPython import embed

def main():
    threshold = 0.0001
    para = 0.5
    alpha = 0.85
    beta = 0.0
    ap = True
    if ap:
        budget = 1000
    else:
        budget = 10
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    ret = {}
    for ii, s in enumerate(duc_set):
        print('Running textrank for %d doc...' % ii)
        passages, raw_sentences = generate_duc_docs(s, stopword_path)
        similarity_scores = calculate_similarity(passages)
        normalized_sim = np.zeros(similarity_scores.shape)
        for i in range(similarity_scores.shape[0]):
            similarity_scores[i, i] = 0.0
            sum_ = np.sum(similarity_scores[:, i])
            if sum_ == 0:
                continue
            for j in range(similarity_scores.shape[0]):
                normalized_sim[j, i] = similarity_scores[j, i] / sum_

        sent_num = len(passages)
        scores = 1.0 / sent_num * np.ones([sent_num])
        current_scores = scores.copy()

        while True:
            #print('Update...')
            scores = alpha * np.dot(normalized_sim, scores) + (1 - alpha)
            dist = np.linalg.norm(current_scores - scores)
            if dist < threshold:
                break
            current_scores = scores

        #ranked_list = []
        #for idx, score in enumerate(scores):
        #    ranked_list.append((idx, scores[idx]))
        #ranked_list = sorted(ranked_list, key=lambda t: -t[1])

        chosen = [False for _ in raw_sentences]
        summary_id = []

        current_len = 0
        while len(summary_id) < budget:
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
                if new_score > maxscore:
                    maxscore = new_score
                    pick = i
            if pick == -1:
                break
            current_len += len(raw_sentences[pick])
            chosen[pick] = True
            summary_id.append(pick)

        #summary = ''
        #for id in summary_id:
        #    summary += raw_sentences[id] + ' '

        l = [-1 for _ in passages]
        if ap:
            for idx, i in enumerate(summary_id):
                l[i] = len(summary_id) - idx
            ret[s] = l
        else:
            for i in summary_id:
                l[i] = 1
            ret[s] = l

        #f = open('/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/tmp/system1/' + s + 't.txt', 'w')
        #f.write(summary)
        #f.close()

    pickle.dump(ret, open('./data/textrank_sentence_res.p', 'wb'))

def calculate_similarity(passages):
    sent_num = len(passages)
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
                sim = len(set(passages[i]) & set(passages[j])) / (math.log(len(set(passages[j])) + 1) + math.log(len(set(passages[j])) + 1))
                similarity_scores[i][j] = sim
    return similarity_scores

if __name__ == '__main__':
    main()