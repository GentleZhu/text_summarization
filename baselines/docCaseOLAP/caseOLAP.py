from doc_clustering import *
from phraseExtractor import *
from relevance_opt import *
from IPython import embed
import numpy as np

sim_low = 0.6
sim_high = 0.85
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

def build_matrix_mixed(phrase_num, phrase2idx, target_passages, window_size, model_path):
    # Build a matrix based on co-occurrence and embedding similarity.
    model = Word2Vec(size=300, min_count=1)
    phrases = list(phrase2idx.keys())
    model.build_vocab([phrases])
    model.intersect_word2vec_format(model_path, binary=False, lockf=1.0)

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
                sim = np.fabs(model.wv.similarity(phrase_i, phrase_j))
                link_weights[phrase2idx[phrase_i]][phrase2idx[phrase_j]] += sim
                link_weights[phrase2idx[phrase_j]][phrase2idx[phrase_i]] += sim

    assert (phrase_num == len(link_weights[0]))
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

def build_matrix_cooccur(phrase_num, phrase2idx, target_passages, window_size):
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

def build_matrix_embedding(model_path, idx2phrase):
    model = Word2Vec(size=300, min_count=1)
    phrases = [idx2phrase[idx] for idx in idx2phrase]
    model.build_vocab([phrases])
    # glove_file = datapath(emb_path)
    # tmp_file = get_tmpfile('tmp.txt')
    # glove2word2vec(glove_file, tmp_file)
    # model.intersect_word2vec_format(tmp_file, binary=False, lockf=1.0)
    model.intersect_word2vec_format(model_path, binary=False, lockf=1.0)

    phrase_num = len(idx2phrase)
    A = np.ones((phrase_num, phrase_num))
    for idx_i in range(phrase_num):
        total_weight = 0
        for idx_j in range(phrase_num):
            if idx_i == idx_j:
                continue
            A[idx_i][idx_j] = np.fabs(model.wv.similarity(idx2phrase[idx_i], idx2phrase[idx_j]))
            #A[idx_i][idx_j] = model.wv.similarity(idx2phrase[idx_i], idx2phrase[idx_j])
            #if A[idx_i][idx_j] < 0:
            #    A[idx_i][idx_j] = 0
            total_weight += A[idx_i][idx_j]

        if total_weight == 0:
            A[idx_i] = np.ones(phrase_num) / phrase_num
        else:
            for idx_j in range(phrase_num):
                A[idx_i][idx_j] = A[idx_i][idx_j] / total_weight

    return A

def textrank(target_passages, phrase_bias, window_size=3, link_option='co', use_bias=False):
    # Link option: co-occur, embedding, mixed
    phrase_num = len(phrase_bias)
    phrase2idx = {}
    idx2phrase = {}
    for idx, phrase in enumerate(phrase_bias):
        phrase2idx[phrase] = idx
        idx2phrase[idx] = phrase
    assert(len(phrase2idx) == phrase_num)
    assert(link_option in ['co', 'emb', 'mixed'])
    if link_option == 'co':
        A = build_matrix_cooccur(phrase_num, phrase2idx, target_passages, window_size)
    elif link_option == 'emb':
        A = build_matrix_embedding('/shared/data/qiz3/text_summ/src/jt_code/finetune_nyt.emb', idx2phrase)
    else:
        A = build_matrix_mixed(phrase_num, phrase2idx, target_passages, window_size,
                               '/shared/data/qiz3/text_summ/src/jt_code/finetune_nyt.emb')
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

if __name__ == '__main__':

    model_path = '/shared/data/qiz3/text_summ/output_data/nyt13_110k_docvecs.npy'
    train_data = np.load(model_path)
    clusterer = kb_doc_clustering(train_data, cluster_number)
    cluster_membership = get_cluster_membership(clusterer)
    embed()
    exit()

    # Each doc id in Target set is indexed from 1, not 0.
    target_set = [1]
    corpusIn = open('/shared/data/qiz3/text_summ/data/nyt13_110k_summ.txt')
    stopword_path = '../../data/stopwords.txt'

    # passages are not further split into sentences.
    # document_phrase_cnt is the cnt of phrases in each document. inverted_index in a reversed way.
    passages, document_phrase_cnt, inverted_index = load_corpus_doc2vec(corpusIn, stopword_path)

    # Calculate the frequency of candidates in target documents.
    target_phrase_freq = calculate_target_phrase_freq(document_phrase_cnt, target_set)

    # When use the whole corpus as background.
    #whole_background = [x for x in range(100) if not x in target_set]
    phrase2idx = {}
    idx2phrase = {}
    for idx, phrase in enumerate(target_phrase_freq):
        phrase2idx[phrase] = idx
        idx2phrase[idx] = phrase

    phrase_num = len(phrase2idx)
    phrase_candidates = [idx2phrase[idx] for idx in range(phrase_num)]

    # Train document embedding and document clustering.
    model = train_doc2vec(passages)
    clusterer = doc_clustering(model, cluster_number)
    cluster_membership = get_cluster_membership(clusterer)

    embed()
    exit()
    
    #S = construct_matrix(phrase_candidates, passages, document_phrase_cnt, inverted_index)
    #f = optimize_relevance(S, target_set, 100, 30, 0.0001)
    #relevance_scores = [f[idx] for idx in range(len(phrase_candidates))]

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

    phrase_extractor = phraseExtractor(phrase_candidates, phrase2idx, target_set, sibling_groups, target_phrase_freq)
    #If you want to get results from tf and distinctiveness, use the following:
    #ranked_list = phrase_extractor.compute_scores(document_phrase_cnt, inverted_index, 'B')
    ranked_list = phrase_extractor.compute_scores(document_phrase_cnt, inverted_index, 'B')

    embed()
    exit()

    phrase_bias = {t[0]: t[1] for t in ranked_list}
    target_passages = [passages[idx - 1] for idx in target_set]
    ranked_list = textrank(target_passages, phrase_bias, use_bias=False)
