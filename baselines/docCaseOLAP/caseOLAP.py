from doc_clustering import *
from phraseExtractor import *
from IPython import embed
import numpy as np

sim_low = 0.6
sim_high = 0.8
cluster_number = 50

def pagerank(A, init_scores, eps=0.0001, d=0.5):
    P = np.ones(A.shape[0]) / A.shape[0]
    #print(P)
    count=0
    while True:
        new_P = init_scores * (1 - d) + d * A.T.dot(P)
        #print(new_P)
        delta = abs(new_P - P).sum()
        count+=1
        if delta <= eps:
            #print(count)
            return new_P
        P = new_P

def build_matrix(link_weights):
    phrase_num = len(link_weights)
    assert(phrase_num == len(link_weights[0]))
    A = np.ones((phrase_num, phrase_num))
    for idx_i in range(phrase_num):
        total_weights = 0
        for idx_j in range(phrase_num):
            total_weights += link_weights[idx_i][idx_j]
        if total_weights == 0:
            A[idx_i] = np.ones(phrase_num) / phrase_num
        else:
            for idx_j in range(phrase_num):
                A[idx_i][idx_j] = link_weights[idx_i][idx_j] / total_weights

    return A

def textrank(target_passages, phrase_bias, window_size=3, use_bias=False):
    phrase_num = len(phrase_bias)
    phrase2idx = {}
    idx2phrase = {}
    for idx, phrase in enumerate(phrase_bias):
        phrase2idx[phrase] = idx
        idx2phrase[idx] = phrase
    assert(len(phrase2idx) == phrase_num)
    link_weights = [[0 for _ in range(phrase_num)] for _ in range(phrase_num)]
    for passage in target_passages:
        passage_len = len(passage)
        for idx_i in range(passage_len):
            for w in range(window_size):
                idx_j = idx_i + w
                if idx_j > passage_len - 1:
                    break
                phrase_i, phrase_j = passage[idx_i], passage[idx_j]
                if phrase_i == phrase_j:
                    continue
                link_weights[phrase2idx[phrase_i]][phrase2idx[phrase_j]] += 1
                link_weights[phrase2idx[phrase_j]][phrase2idx[phrase_i]] += 1
    A = build_matrix(link_weights)
    print('Matrix built.')
    init_score = np.ones(phrase_num) / phrase_num
    if use_bias:
        init_score = [0 for _ in range(phrase_num)]
        for phrase in phrase_bias:
            init_score[phrase2idx[phrase]] = phrase_bias[phrase]
        init_score = np.array(init_score) / np.sum(init_score)
    P = pagerank(A, init_score)
    ranked_list = [(idx2phrase[idx], P[idx]) for idx in range(phrase_num)]
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    return ranked_list

# Each doc id in Target set is indexed from 1, not 0.
target_set = [31,32,41,42]
corpusIn = open('../../intermediate_data/nyt13_10k_9_25.txt')
stopword_path = '../../data/stopwords.txt'

# passages are not further split into sentences.
# document_phrase_cnt is the cnt of phrases in each document. inverted_index in a reversed way.
passages, document_phrase_cnt, inverted_index = load_corpus_doc2vec(corpusIn, stopword_path)

# Calculate the frequency of candidates in target documents.
target_phrase_freq = calculate_target_phrase_freq(document_phrase_cnt, target_set)

# When use the whole corpus as background.
whole_background = [x for x in range(len(passages)) if not x in target_set]
phrase_candidates = set()
for doc_name in ['a_' + str(x) for x in target_set]:
    for phrase in document_phrase_cnt[doc_name]:
        if not phrase in phrase_candidates:
            phrase_candidates.add(phrase)

# Train document embedding and document clustering.
model = train_doc2vec(passages)
clusterer = doc_clustering(model, cluster_number)
cluster_membership = get_cluster_membership(clusterer)

# Here, doc_id has no 'a_' as prefix.
# For each cluster, we pick those documents that are similar to target as one sibling group.
# We eliminate those documents above a similarity threshold from sibling group.
sibling_groups = []
for cluster_idx in range(cluster_number):
    cluster_members = cluster_membership[cluster_idx]
    ranked_list = [int(doc_id[0][2:]) for doc_id in rank_docs_similarity(target_set, cluster_members, model, reverse=True)
                   if doc_id[1] > sim_low and doc_id[1] < sim_high]
    if len(ranked_list) > 0:
        sibling_groups.append(ranked_list)

phrase_extractor = phraseExtractor(list(phrase_candidates), target_set, sibling_groups, target_phrase_freq)
#If you want to get results from tf and distinctiveness, use the following:
# ranked_list = phrase_extractor.compute_scores(document_phrase_cnt, inverted_index, True)
ranked_list = phrase_extractor.compute_scores(document_phrase_cnt, inverted_index, False)

embed()

phrase_bias = {t[0]: t[1] for t in ranked_list}

target_passages = [passages[idx - 1] for idx in target_set]
ranked_list = textrank(target_passages, phrase_bias, use_bias=True)

embed()
exit()
