from IPython import embed
import numpy as np
import sys
from tqdm import tqdm
import random
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from model import GCN
import torch
import torch.nn.functional as F
sys.path.append('../')

from collections import defaultdict
import pickle, scipy, nltk
import re, math
from numpy.linalg import matrix_power

from phraseExtractor import phraseExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def GCNRanker(target_phrases, similarity_scores, phrase2idx, idx2phrase, labels, hidden = 16):
    assert np.allclose(similarity_scores, similarity_scores.T)
    for i in range(similarity_scores.shape[0]):
        similarity_scores[i,i] = similarity_scores[i,i] + 1.0
    rowsum = 1.0 / similarity_scores.sum(1)
    
    normalized_sim = np.diag(rowsum).dot(similarity_scores).astype(np.float32)
    #print(normalized_sim.sum(1))
    # Load data
    input_labels = np.zeros([similarity_scores.shape[0]])
    idx_train = []
    for p in labels:
        if labels[p] == 1:
            input_labels[phrase2idx[p]] = 1
        idx_train.append(phrase2idx[p])

    idx_train = torch.LongTensor(idx_train)
    labels = torch.LongTensor(input_labels)
    features = torch.from_numpy(np.ones([similarity_scores.shape[0], 8], dtype=np.float32))
    adj = torch.from_numpy(normalized_sim)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(),
                           lr=0.001)

    model.train()
    

    for epoch in range(200):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print("Epoch:{}, Loss:{}".format(epoch, loss_train))
    print(labels[idx_train])
    model.eval()
    output = model(features, adj)
    preds = output.max(1)[1].type_as(labels)
    top_list = []
    for i in range(len(phrase2idx)):
        #if preds[i] == 1:
        top_list.append([idx2phrase[i], math.exp(output[i][1].data.cpu().item())])
    top_list = sorted(top_list, key=lambda x:x[1], reverse=True)
    print(top_list[:30])
    return top_list

def seedRanker(target_phrases, similarity_scores, phrase2idx, idx2phrase, labels):
    # Textrank.
    threshold = 0.0001
    alpha = 0.85

    normalized_sim = np.zeros(similarity_scores.shape)
    for i in range(similarity_scores.shape[0]):
        similarity_scores[i, i] = 0.0
        sum_ = np.sum(similarity_scores[:, i])
        if sum_ == 0:
            continue
        for j in range(similarity_scores.shape[0]):
            normalized_sim[j, i] = similarity_scores[j, i] / sum_

    num_target = len(target_phrases)
    # together with embedding learning
    assert num_target == similarity_scores.shape[0]
    I = np.eye(num_target)
    weight = np.zeros([num_target])
    for t in target_phrases:
        if t in labels:
            if labels[t] == 0:
                I[phrase2idx[t],phrase2idx[t]] = 0
            elif labels[t] == 1:
                weight[phrase2idx[t]] = 1
    scores = 1.0 / num_target * np.ones([num_target])
    #topic_scores = scores.copy()
    current_scores = scores.copy()
    #if len(sink) > 0:
    #    print(sum(sum(I)), num_target)
    while True:
        #print('Update...')
        scores = alpha * np.dot(np.matmul(normalized_sim, I), scores) + (1 - alpha) * weight
        dist = np.linalg.norm(current_scores - scores)
        if dist < threshold:
            break
        current_scores = scores
    ranked_list = [(idx2phrase[i], score) for (i, score) in enumerate(scores)]
    ranked_list = sorted(ranked_list, key=lambda t:-t[1])
    return ranked_list

def manifold_ranking(twin_phrases, target_phrases, topic_scores, phrase2idx, similarity_scores):
    # Manifold ranking with sink points.
    # The phrases in twin sets are regarded as sink points, and ranked simultaneously.
    threshold = 0.0001
    alpha = 0.7

    normalized_sim = np.zeros(similarity_scores.shape)
    for i in range(similarity_scores.shape[0]):
        similarity_scores[i, i] = 0.0
        sum_ = np.sum(similarity_scores[:, i])
        if sum_ == 0.0:
            continue
        for j in range(similarity_scores.shape[0]):
            normalized_sim[j, i] = similarity_scores[j, i] / sum_

    num_twin = len(twin_phrases)
    num_target = len(target_phrases)
    I_f = np.zeros([num_twin + num_target, num_twin + num_target])
    for phrase in target_phrases:
        I_f[phrase2idx[phrase], phrase2idx[phrase]] = 1.0
    #I_f = np.eye(num_twin + num_target)
    topic_scores = 1.0 / np.sum(topic_scores) * topic_scores# np.ones([num_target + num_twin])
    scores = 1.0 / (num_target + num_twin) * np.ones([num_target + num_twin])
    #topic_scores = scores.copy()
    current_scores = scores.copy()

    while True:
        print('Update...')
        scores = alpha * np.dot(normalized_sim, scores) + (1 - alpha) * topic_scores
        dist = np.linalg.norm(current_scores - scores)
        if dist < threshold:
            break
        current_scores = scores

    return scores

def ensembleSumm(ranklist, k, opt=0):
    final_ranking = defaultdict(float)
    if opt == 0:
        for r in ranklist:
            for i in range(k):
                final_ranking[r[i][0]] += 1.0 / (i+1)
    return sorted(final_ranking.items(), key=lambda x:x[1], reverse=True)[:k]

def distinctScore():
    pass


def textrank(target_phrases, similarity_scores, reweight=None, sink=[]):
    # Textrank.
    threshold = 0.0001
    alpha = 0.85

    normalized_sim = np.zeros(similarity_scores.shape)
    for i in range(similarity_scores.shape[0]):
        similarity_scores[i, i] = 0.0
        sum_ = np.sum(similarity_scores[:, i])
        if sum_ == 0:
            continue
        for j in range(similarity_scores.shape[0]):
            normalized_sim[j, i] = similarity_scores[j, i] / sum_

    num_target = len(target_phrases)
    # together with embedding learning
    assert num_target == similarity_scores.shape[0]
    I = np.eye(num_target)
    for idx,t in enumerate(target_phrases):
            if idx in sink:
                I[idx,idx] = 0

    weight = np.ones([num_target])
    if reweight:
        for idx,t in enumerate(target_phrases):
            if t in reweight:
                weight[idx] = reweight[t] 
    
    scores = 1.0 / num_target * np.ones([num_target])
    #topic_scores = scores.copy()
    current_scores = scores.copy()
    #if len(sink) > 0:
    #    print(sum(sum(I)), num_target)
    while True:
        #print('Update...')
        scores = alpha * np.dot(np.matmul(normalized_sim, I), scores) + (1 - alpha) * weight
        dist = np.linalg.norm(current_scores - scores)
        if dist < threshold:
            break
        current_scores = scores

    return scores

def build_co_occurrence_matrix(target, phrase2idx, seg_file):
    segIn = open(seg_file)
    similarity_scores = np.zeros([len(phrase2idx), len(phrase2idx)])
    doc_id = -1
    passage = []
    IN_PHRASE_FLAG = False
    for cline in tqdm(segIn):
        doc_id += 1
        if doc_id not in target:
            continue
        cline = cline.replace('<phrase>', '<phrase> ').replace('</phrase>', ' </phrase>').replace('\n', '')
        words = cline.split(' ')
        tmp_passage = []
        token_list = []
        for t in words:
            t = t.lower()
            if t == '<phrase>':
                token_list = []
                IN_PHRASE_FLAG = True
            elif t == '</phrase>':
                IN_PHRASE_FLAG = False
                tmp_passage.append('_'.join(token_list))
            elif IN_PHRASE_FLAG:
                token_list.append(t)
        '''
        tmp_similarity_scores = np.zeros([len(phrase2idx), len(phrase2idx)])
        single_occ_threshold = 3
        for idx, phrase in enumerate(tmp_passage):
            if phrase not in phrase2idx:
                continue
            for w in range(1, 4):
                if idx + w > len(sentence) - 1:
                    break
                if sentence[idx + w] not in phrase2idx:
                    continue
                tmp_similarity_scores[phrase2idx[phrase], phrase2idx[sentence[idx + w]]] += 1
                tmp_similarity_scores[phrase2idx[sentence[idx + w]], phrase2idx[phrase]] += 1
                if tmp_similarity_scores[phrase2idx[phrase], phrase2idx[sentence[idx + w]]] > single_occ_threshold:
                    tmp_similarity_scores[phrase2idx[phrase], phrase2idx[sentence[idx + w]]] = single_occ_threshold
                if tmp_similarity_scores[phrase2idx[sentence[idx + w]], phrase2idx[phrase]] > single_occ_threshold:
                    tmp_similarity_scores[phrase2idx[sentence[idx + w]], phrase2idx[phrase]] = single_occ_threshold
        similarity_scores += tmp_similarity_scores
    '''
        passage.append(tmp_passage)
    for sentence in passage:
        for idx, phrase in enumerate(sentence):
            if phrase not in phrase2idx:
                continue
            for w in range(1, 5):
                if idx + w > len(sentence) - 1:
                    break
                if sentence[idx + w] not in phrase2idx:
                    continue
                similarity_scores[phrase2idx[phrase], phrase2idx[sentence[idx + w]]] += 1
                similarity_scores[phrase2idx[sentence[idx + w]], phrase2idx[phrase]] += 1
    #'''
    return similarity_scores

def build_target_co_occurrence_matrix(target_docs, phrase2idx):
    similarity_scores = np.zeros([len(phrase2idx), len(phrase2idx)])
    for sentence in target_docs:
        for idx, phrase in enumerate(sentence):
            if phrase not in phrase2idx:
                continue
            for w in range(1, 5):
                if idx + w > len(sentence) - 1:
                    break
                if sentence[idx + w] not in phrase2idx:
                    continue
                similarity_scores[phrase2idx[phrase], phrase2idx[sentence[idx + w]]] += 1
                similarity_scores[phrase2idx[sentence[idx + w]], phrase2idx[phrase]] += 1
    #'''
    return similarity_scores

def leven_similarity(phrase_a, phrase_b):
    # Also known as 'edit distance'
    len1, len2 = len(phrase_a), len(phrase_b)
    #if not (phrase_a[:5] == phrase_b[:5] or phrase_a[-5:] == phrase_b[-5:]):
    #    return 10
    matrix = [[0 for _ in range(len2)] for _ in range(len1)]
    for i in range(len1):
        for j in range(len2):
            if i == 0 or j == 0:
                matrix[i][j] = max(i, j)
            else:
                if phrase_b[j] != phrase_a[i]:
                    matrix[i][j] = min([matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + 1])
                else:
                    matrix[i][j] = min([matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1]])
    return matrix[len1 - 1][len2 - 1]

def select_phrases(relevance_score, similarity_score, weight, k):
    # Diversified ranking.
    # relevance score: 1-d numpy array. similarity score: 2-d numpy array
    # weight: >= 2 to ensure monotonity.
    reference_vector = np.dot(relevance_score, similarity_score)
    #reference_vector = np.ones(relevance_score.shape[0])
    score = weight * (reference_vector * relevance_score) - np.diag(similarity_score) * relevance_score * relevance_score
    selected_indices = list()
    for i in range(k):
        current_max = -1
        ret = 0
        for idx, x in enumerate(score):
            if idx in selected_indices:
                continue
            if x > current_max:
                current_max = x
                ret = idx
        selected_indices.append(ret)
        score = score - 2 * relevance_score[ret] * (similarity_score[:, ret] * relevance_score)
    return selected_indices

def collect_statistics(in_file):
    document_phrases = defaultdict(lambda: defaultdict(int))
    inverted_index = defaultdict(lambda: defaultdict(int))
    with open(in_file) as IN:
        for line in IN:
            line = line.strip('\n').split('\t')
            if line[1] != '':
                doc_id, phrases = int(line[0]), line[1].split(';')
            else:
                continue
            for phrase in phrases:
                document_phrases[doc_id][phrase] += 1
                inverted_index[phrase][doc_id] += 1
    return document_phrases, inverted_index

def load_doc_sets(sib_file='/shared/data/qiz3/text_summ/src/jt_code/doc2cube/src/sib.dump',
                        twin_file='/shared/data/qiz3/text_summ/src/jt_code/doc2cube/src/twin.dump'):
    siblings = pickle.load(open(sib_file, 'rb'))
    twin_docs = pickle.load(open(twin_file, 'rb'))
    return siblings, twin_docs

def generate_candidate_phrases(document_phrase_cnt, docs):
    # Given doc_ids, generate all candidate phrases.
    phrase2idx = defaultdict()
    for doc in docs:
        for phrase in doc:
            if phrase not in phrase2idx:
                phrase2idx[phrase] = len(phrase2idx)
    return phrase2idx

def generate_caseOLAP_scores(sibling_groups, target_docs, document_phrase_cnt, inverted_index, phrase2idx, option):
    phrase_candidates = list(phrase2idx.keys())
    target_phrase_freq = defaultdict(int)
    for doc in target_docs:
        for phrase in doc:
            target_phrase_freq[phrase] += 1

    phrase_extractor = phraseExtractor(phrase_candidates, phrase2idx, target_docs, sibling_groups, target_phrase_freq)
    ranked_list = phrase_extractor.compute_scores(document_phrase_cnt, inverted_index, option)
    scores = np.array([0.0 for _ in range(len(ranked_list))])

    for t in ranked_list:
        scores[phrase2idx[t[0]]] = t[1]
    return scores,ranked_list

def calculate_pairwise_similarity(phrase2idx):
    # phrase2idx: dict, {'USA':1, ... }
    idx2phrase = {phrase2idx[k]:k for k in phrase2idx}
    similarity_scores = np.zeros([len(phrase2idx), len(phrase2idx)])
    emb = load_emb('/shared/data/qiz3/text_summ/src/jt_code/full.emb')
    print('Calculate pairwise similarity...')
    for i in tqdm(range(len(phrase2idx))):
        for j in range(len(phrase2idx)):
            if j < i:
                similarity_scores[i][j] = similarity_scores[j][i]
            else:
                #similarity_scores[i][j] = 1.0 / (1 + leven_similarity(idx2phrase[i], idx2phrase[j]))
                similarity_scores[i][j] = calc_sim(emb, idx2phrase[i], idx2phrase[j])
    #similarity_scores = np.zeros([len(phrase2idx), len(phrase2idx)])
    #similarity_scores = np.eye(len(phrase2idx))
    return similarity_scores, idx2phrase

def contrastive_analysis(document_phrase_cnt, background_phrases, twin, target):
    # Rerank phrases in target docs using twin docs.
    # Currently use freq diff.
    twin_cnt = defaultdict(int)
    target_cnt = defaultdict(int)
    n = defaultdict(int)
    for doc_id in twin:
        for phrase in document_phrase_cnt[doc_id]:
            twin_cnt[phrase] += document_phrase_cnt[doc_id][phrase]
    for doc_id in target:
        for phrase in document_phrase_cnt[doc_id]:
            n[phrase] += 1
            target_cnt[phrase] += document_phrase_cnt[doc_id][phrase]
    phrase_rescore = {}
    for phrase in background_phrases:
        if phrase not in target_cnt:
            continue
        phrase_rescore[phrase] = 1.0 * target_cnt[phrase] / twin_cnt[phrase] * len(twin) / len(target)
    ranked_list = [(phrase, phrase_rescore[phrase]) for phrase in phrase_rescore]
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    return ranked_list, phrase_rescore

def graph_optimization(target_candidates, background_candidates, phrase2idx, target_saliency, background_saliency,
                       relevance_matrix):
    num_concept = len(phrase2idx)
    max_iter = 10
    out_iter = 5
    convergence_threshold = 0.01
    gamma = 0.1
    alpha = 1.0
    lambda_ = 1

    concept_list = list(phrase2idx.keys())
    f_doc_target = np.zeros(num_concept)
    f_doc_bg = np.zeros(num_concept)
    f_target = np.zeros(num_concept)
    f_bg = np.zeros(num_concept)
    y_target = np.zeros(num_concept)
    g_target = np.zeros(num_concept)
    g_bg = np.zeros(num_concept)
    for phrase in target_saliency:
        g_target[phrase2idx[phrase]] = target_saliency[phrase]
    for phrase in background_saliency:
        g_bg[phrase2idx[phrase]] = target_saliency[phrase]

    iter_outter = 0
    obj = -1.0
    obj_old_outter = obj
    while iter_outter < out_iter:
        iter_inner = 0
        obj_old = obj
        while iter_inner < max_iter:
            # Update f
            f_target = f_target # TODO
            f_bg = f_bg # TODO

            L1 = 2 * f_target.dot(f_target) - 2 * f_target.dot(relevance_matrix * f_target)
            L2 = 2 * f_bg.dot(f_bg) - 2 * f_bg.dot(relevance_matrix * f_bg)
            distinct_score = 0.
            for i in range(num_concept):
                distinct_score += y_target[i] * math.log((f_target[i] + gamma) / (f_bg[i] + gamma))
            obj = 0.5 * (L1 + L2 + alpha * np.linalg.norm(f_target - g_target) ** 2
                         + alpha * np.linalg.norm(f_target - g_target) ** 2) - lambda_ * distinct_score

            iter_inner += 1
            obj_rel = math.fabs(obj_old - obj) / math.fabs(obj_old)
            if obj_rel < convergence_threshold:
                break

        for i in range(num_concept):
            f_doc_target[i] = math.log((f_target[i] + gamma) / (f_bg[i] + gamma))
            f_doc_bg[i] = math.log((f_bg[i] + gamma) / (f_target[i] + gamma))
        y_target_threshold = 0.0
        y_bg_sup = len(background_candidates)
        for phrase in background_candidates:
            y_target_threshold += f_doc_bg[phrase2idx[phrase]] / y_bg_sup
        y_bg_threshold = 0.0
        y_target_sup = len(target_candidates)
        for phrase in target_candidates:
            y_bg_threshold += f_doc_target[phrase2idx[phrase]] / y_target_sup

        y_target = np.zeros(num_concept)
        for phrase in target_candidates:
            if f_doc_target[phrase2idx[phrase]] > y_target_threshold:
                y_target[phrase2idx[phrase]] = 1.0

        iter_outter += 1
        obj_rel_outter = math.fabs(obj_old_outter - obj) / math.fabs(obj_old_outter)
        if obj_rel_outter < convergence_threshold:
            break

    id2phrase = {phrase2idx[k]: k for k in phrase2idx}
    chosen_phrases = [id2phrase[i] for i in range(num_concept) if y_target[i] == 1.0]
    return chosen_phrases

def random_sample_sibling(siblings, k):
    # Randomly sample k docs for each sibling.
    new_siblings = []
    for sibling in siblings:
        if len(sibling) <= k:
            new_siblings.append(sibling)
        else:
            new_siblings.append(random.sample(sibling, k))
    return new_siblings

def calc_sim(emb, phrase_a, phrase_b):
    vec_a = emb[phrase_a] if phrase_a in emb else emb['(']
    vec_b = emb[phrase_b] if phrase_b in emb else emb['(']
    return 1 - scipy.spatial.distance.cosine(vec_a, vec_b)

def sub_modular(emb, candidate_phrases, k, weight):
    ###########################
    # Still buggy, not tested #
    ###########################
    selected = set()
    unselected = set(candidate_phrases)
    current_score = 0.
    for i in range(k):
        current_max = -1
        choice = ''
        for phrase in candidate_phrases:
            if phrase in selected:
                continue
            tmp_score = current_score
            for phrase_ in selected:
                tmp_score -= calc_sim(emb, phrase, phrase_)
                tmp_score -= weight * calc_sim(emb, phrase, phrase_)
            for phrase_ in unselected:
                if phrase == phrase_:
                    continue
                tmp_score += calc_sim(emb, phrase, phrase_)
            if tmp_score > current_max:
                current_max = tmp_score
                choice = phrase
        if current_max > current_score:
            current_score = current_max
            selected.add(choice)
            unselected -= choice
        else:
            break
    return selected

def load_emb(emb_path):
    emb = {}
    with open(emb_path) as IN:
        line_n = 0
        for line in IN:
            if line_n == 0:
                line_n = 1
                continue
            data = line.split(' ')
            phrase, em = data[0], data[1:]
            em = [float(x) for x in em]
            emb[phrase] = em
    return emb

def load_segmented_corpus(segIn, stopword_path):
    doc_id = 0
    passages = []
    stopwords = set()
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())
    IN_PHRASE_FLAG = False
    for cline in tqdm(segIn):
        doc_id += 1
        cline = cline.replace('<phrase>', '<phrase> ').replace('</phrase>', ' </phrase>').replace('\n', '')
        words = cline.split(' ')
        tmp_passage = []
        token_list = []
        for t in words:
            t = t.lower()
            if t == '<phrase>':
                token_list = []
                IN_PHRASE_FLAG = True
            elif t == '</phrase>':
                IN_PHRASE_FLAG = False
                tmp_passage.append('_'.join(token_list))
            elif IN_PHRASE_FLAG:
                token_list.append(t)
            else:
                if t in stopwords or t in ['.', ',', '?', '!']:
                    continue
                tmp_passage.append(t)
        passages.append(tmp_passage)
    return passages

def train_doc2vec(passages):
    tagged_data = [TaggedDocument(words=d_, tags=[str(i)]) for i, d_ in enumerate(passages)]
    model = Doc2Vec(tagged_data, dm=1, size=100, window=5, min_count=1, workers=10)
    return model

def rank_phrase_emb(phrase2idx, model, avg_doc):
    ranked_list = []
    for phrase in phrase2idx:
        if phrase not in model.wv.vocab:
            ranked_list.append((phrase, 0.0))
        v = model.wv[phrase]
        sim = 1 - scipy.spatial.distance.cosine(v, avg_doc)
        ranked_list.append((phrase, sim))
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    return ranked_list

def build_in_domain_dict(target_docs, document_phrase_cnt):
    #out of vocabulary phrases stat
    phrase2idx = generate_candidate_phrases(document_phrase_cnt, target_docs)
    idx2phrase = {phrase2idx[k]: k for k in phrase2idx}
    return phrase2idx, idx2phrase

def pick_sentences(phrase_scores, budget, passage):
    lambda_ = 0.5
    c = 0.5
    sentences = nltk.tokenize.sent_tokenize(passage)
    sent_scores = {}
    sent_phrases = defaultdict(set)
    for idx, sentence in enumerate(sentences):
        phrases = re.findall('<phrase>.*?</phrase>', sentence)
        phrases = [phrase[8:-9].replace(' ', '_').lower() for phrase in phrases]
        sent_phrases[idx] = set(phrases)
        score = sum([phrase_scores[phrase] for phrase in phrases])
        sent_scores[idx] = score

    chosen = list()
    current_len = 0
    current_p = set()
    while current_len < budget:
        max_ = -1
        max_idx = -1
        for idx in range(len(sent_phrases)):
            if idx in chosen or current_len + len(sentences[idx]) > budget:
                continue
            score_gain = sum([phrase_scores[t] for t in sent_phrases[idx]]) + lambda_ * len(set(sent_phrases[idx]) - current_p)
            score_gain /= math.pow(len(sentences[idx].replace('<phrase>', '').replace('</phrase>', '')), c)
            if score_gain > max_:
                max_ = score_gain
                max_idx = idx
        if max_idx != -1:
            chosen.append(max_idx)
            current_len += len(sentences[max_idx])
            current_p |= sent_phrases[max_idx]
        else:
            break
    return sentences, chosen

def main():
    target_docs = [846, 845, 2394, 2904, 2633, 2565, 2956, 2728, 2491]
    #target_docs = [846]
    segIn = open('/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt')
    stopword_path = '../../data/stopwords.txt'
    #passages = load_segmented_corpus(segIn, stopword_path)
    document_phrase_cnt, inverted_index = collect_statistics('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/full.txt')
    siblings, twin_docs = load_doc_sets()

    phrase2idx, idx2phrase = build_in_domain_dict(target_docs, document_phrase_cnt)

    ####################
    # EmbedRank block ##
    ####################
    model = train_doc2vec(passages)
    avg_vec = np.mean([model.docvecs[str(i)] for i in target_docs], axis=0)
    ranked_list = rank_phrase_emb(phrase2idx, model, avg_vec)
    embed()
    exit()

    ###################
    # Textrank block ##
    ###################
    '''
    similarity_scores = build_co_occurrence_matrix(target_docs, phrase2idx, 
            '/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/sports/intermediate/segmentation.txt')
    scores = textrank(phrase2idx.keys(), similarity_scores)
    ranked_list = [(idx2phrase[i], score) for (i, score) in enumerate(scores)]
    ranked_list = sorted(ranked_list, key=lambda t:-t[1])
    embed()
    exit()
    '''

    #################
    # Unused block ##
    #################
    #similarity_scores = pickle.load(open('similarity_score.p', 'rb'))
    #idx2phrase = pickle.load(open('idx2phrase.p', 'rb'))
    #siblings = random_sample_sibling(siblings, 10)

    ##################
    # caseOLAP block #
    ##################
    '''
    scores, ranked_list = generate_caseOLAP_scores(siblings, target_docs, document_phrase_cnt, inverted_index, phrase2idx)
    embed()
    exit()
    '''

    #################
    # Unused block ##
    #################
    #pickle.dump(ranked_list, open('data/ranked_list.p', 'wb'))
    #sentences, choice = pick_sentences({t[0]: t[1] for t in ranked_list}, 200, passage)


    ##########################
    # Manifold ranking block #
    ##########################
    '''
    scores, ranked_list = generate_caseOLAP_scores(siblings, twin_docs, document_phrase_cnt, inverted_index, phrase2idx)
    phrase_selected = 1000
    all_phrases = [t[0] for t in ranked_list[:phrase_selected]]
    phrase2idx = {phrase: i for (i, phrase) in enumerate(all_phrases)}

    similarity_scores, _ = calculate_pairwise_similarity(phrase2idx)

    topic_scores = np.zeros([phrase_selected])
    for i in range(phrase_selected):
        topic_scores[phrase2idx[ranked_list[i][0]]] = ranked_list[i][1]
    target_phrase2idx = generate_candidate_phrases(document_phrase_cnt, target_docs)
    target_phrases = [phrase for phrase in phrase2idx if phrase in target_phrase2idx]
    twin_phrases = [phrase for phrase in phrase2idx if phrase not in target_phrase2idx]
    A = manifold_ranking(twin_phrases, target_phrases, topic_scores, phrase2idx, similarity_scores)
    ranked_list = [(phrase, A[phrase2idx[phrase]]) for phrase in target_phrases]
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])

    embed()
    exit()
    '''

    selected_index = select_phrases(scores, similarity_scores, 2, 1000)
    phrases = [idx2phrase[k] for k in selected_index]
    #ranked_list, phrase_rescore = contrastive_analysis(document_phrase_cnt, phrases, twin_docs, target_docs)
    embed()
    exit()



def search_nearest_doc(feature_vectors, vectorizer, target_docs, skip_doc = None, contain_doc = None):
    
    new_vectors = vectorizer.transform(target_docs)
    print(target_docs, new_vectors.shape)
    #embed()
    ranked_list = []

    tmp = np.zeros(feature_vectors.shape[0])
    for j in range(new_vectors.shape[0]):
        tmp += linear_kernel(new_vectors[j:j+1], feature_vectors).flatten()
        #tmp_v.append(spatial.distance.cosine(new_vectors[j:j+1], feature_vectors[i].todense()))
        #tmp_v.append(1 - np.dot(new_vectors[j], feature_vectors[i].toarray().squeeze()))
        #embed()
    
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    topk = tmp.argsort()[::-1].tolist()
    for i in topk:
        if skip_doc and i in skip_doc:
            continue
        if contain_doc is None or i in contain_doc:
            ranked_list.append(i)
            if len(ranked_list) >= len(target_docs):
                break
    return ranked_list

def compare(config, docs, feature_vectors, vectorizer, skip_doc = None, contain_doc = None):
    return search_nearest_doc(feature_vectors, vectorizer, docs, skip_doc, contain_doc)

def summary(config, docs, siblings_docs, twin_docs, document_phrase_cnt, inverted_index):
    

    #TODO: @jingjing, rewrite target_doc_assign in utils, you can have label2emb.keys instead call concepts
    '''
    doc_embeddings = model.doc_embeddings()
    hierarchy = simple_hierarchy()
    label, all_siblings = target_hier_doc_assign(hierarchy, docs, label2emb, doc_embeddings, option='hard')
    print(label, all_siblings)
    
    print("Number of sibling groups: {}".format(len(siblings_docs)))

    ranked_list = []
    '''

    if config['summ_method'] == 'caseOLAP':
        phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
        scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index,
                                                       phrase2idx, option='A')
        embed()
        #phrase_scores[duc_set[idx]] = {t[0]: t[1] for t in ranked_list}

    elif config['summ_method'] == 'caseOLAP-twin': 

        phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
        scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt,
                                                       inverted_index, phrase2idx, option='A')

        phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
        scores_, ranked_list_ = generate_caseOLAP_scores([twin_docs], docs, document_phrase_cnt, inverted_index,
                                                       phrase2idx, option='B')
        phrase_scores = {t[0]: t[1] for t in ranked_list_}
        background_scores = {t[0]: t[1] for t in ranked_list}
        for phrase in phrase2idx:
            phrase_scores[phrase] *= background_scores[phrase]
        ranked_list = [(k, phrase_scores[k]) for k in phrase_scores]
        ranked_list = sorted(ranked_list, key=lambda t: -t[1])
        embed()

    elif config['summ_method'] == 'textrank':
        ###################
        # Textrank block ##
        ###################

        phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
        similarity_scores = build_target_co_occurrence_matrix(docs, phrase2idx)
        scores = textrank(phrase2idx.keys(), similarity_scores)
        ranked_list = [(idx2phrase[i], score) for (i, score) in enumerate(scores)]
        ranked_list = sorted(ranked_list, key=lambda t:-t[1])
        embed()
        #phrase_scores[duc_set[idx]] = {t[0]: t[1] for t in ranked_list}

    elif config['summ_method'] == 'kams':

        phrase2idx, idx2phrase = build_in_domain_dict(twin_docs, document_phrase_cnt)
        scores_, ranked_list_ = generate_caseOLAP_scores(siblings_docs, twin_docs, document_phrase_cnt,inverted_index, phrase2idx, option='A')
        phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
        #scores, ranked_list = generate_caseOLAP_scores([twin_docs], docs, document_phrase_cnt, inverted_index, phrase2idx, option='B')
        scores, ranked_list = generate_caseOLAP_scores([twin_docs], docs, document_phrase_cnt, inverted_index, phrase2idx, option='B')
        
        phrase_scores_ = [t[0] for t in ranked_list]
        background_scores = [t[0] for t in ranked_list_]
        
        labels = dict()
        for r in phrase_scores_[:10]:
            labels[r] = 1

        for p in background_scores[:1000]:
            if p in phrase_scores_[:50]:
                labels[p] = 0

        '''
        target_location, twin_location = {}, {}
        for idx,ph in enumerate(ranked_list):
            target_location[ph[0]] = idx
        for idx,ph in enumerate(ranked_list_):
            if ph[0] in target_location:
                twin_location[ph[0]] = len(twin_location)
        location_diff = {}
        '''


        similarity_scores = build_co_occurrence_matrix(docs, phrase2idx,
                '/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt')
        
        ranked_list = seedRanker(phrase2idx.keys(), similarity_scores, phrase2idx, idx2phrase, labels)
        #kams_ranked_list = GCNRanker(phrase2idx.keys(), similarity_scores, phrase2idx, idx2phrase, labels)
        #kams_ranked_list = sorted(kams_ranked_list, key=lambda x:x[1], reverse=True)

        #embed()
        #phrase_scores[duc_set[idx]] = {t[0]: t[1] for t in ranked_list}

        '''
        ##########################
        # Manifold ranking block #
        ##########################
        '''
        '''
        phrase2idx, idx2phrase = build_in_domain_dict(twin_docs, document_phrase_cnt)
        scores, ranked_list = generate_caseOLAP_scores(siblings_docs, twin_docs, document_phrase_cnt,
                                                       inverted_index,
                                                       phrase2idx)
        phrase_selected = 1000
        all_phrases = [t[0] for t in ranked_list[:phrase_selected]]
        phrase2idx = {phrase: i for (i, phrase) in enumerate(all_phrases)}
        idx2phrase = {phrase2idx[k]: k for k in phrase2idx}
        similarity_scores, _ = calculate_pairwise_similarity(phrase2idx)
        topic_scores = np.zeros([len(phrase2idx)])
        for i in range(phrase_selected):
            topic_scores[phrase2idx[ranked_list[i][0]]] = ranked_list[i][1]
        target_phrase2idx = generate_candidate_phrases(document_phrase_cnt, docs)
        target_phrases = [phrase for phrase in phrase2idx if phrase in target_phrase2idx]
        twin_phrases = [phrase for phrase in phrase2idx if phrase not in target_phrase2idx]
        A = manifold_ranking(twin_phrases, target_phrases, topic_scores, phrase2idx, similarity_scores)
        ranked_list = [(phrase, A[phrase2idx[phrase]]) for phrase in target_phrases]
        ranked_list = sorted(ranked_list, key=lambda t: -t[1])
        embed()
        '''


        '''
        ranked_lists = []
        for doc in docs:
            phrase2idx, idx2phrase = build_in_domain_dict([doc], document_phrase_cnt)
            similarity_scores = build_co_occurrence_matrix([doc], phrase2idx,
                    '/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt')
            scores = textrank(phrase2idx.keys(), similarity_scores)
            sub_ranked_list = [(idx2phrase[i], score) for (i, score) in enumerate(scores)]
            sub_ranked_list = sorted(sub_ranked_list, key=lambda t:-t[1])
            ranked_lists.append(sub_ranked_list)
        
        #print(doc, ranked_list[:20], rank_in_twins)
        #print('**'.join(list(map(lambda x:x[0], ranked_list[:10]))))
        #embed()

        ranked_list = ensembleSumm(ranked_lists, k=30)
        '''

if __name__ == '__main__':
    main()