from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from collections import defaultdict
import re
import json
from tqdm import tqdm
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from spherecluster import SphericalKMeans
from string import punctuation
from IPython import embed

def load_corpus_doc2vec(corpusIn, stopword_path):
    doc_id = 0
    passages = []
    stopwords = set()
    document_phrase_cnt = defaultdict(lambda: defaultdict(int))
    inverted_index = defaultdict(lambda: defaultdict(int))
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())
    for doc_idx, cline in tqdm(enumerate(corpusIn)):
        doc_id += 1
        words = word_tokenize(cline)
        tmp_passage = []
        for t in words:
            t = re.sub(r'\W+', '', t)
            if t.lower() not in stopwords:
                _pos = wn.synsets(t)
                if len(_pos) > 0:
                    _pos = _pos[0].pos()
                # print(_pos)
                if _pos == 'n' or _pos == 'a' or '_' in t:
                    tmp_passage.append(t.lower())
                    document_phrase_cnt['a_' + str(doc_idx + 1)][t.lower()] += 1
                    inverted_index[t.lower()]['a_' + str(doc_idx + 1)] += 1
        passages.append(tmp_passage)
    return passages, document_phrase_cnt, inverted_index

def train_doc2vec(passages):
    tagged_data = [TaggedDocument(words=d_, tags=['a_' + str(i + 1)]) for i, d_ in enumerate(passages)]
    model = Doc2Vec(tagged_data, dm=1, size=100, window=5, min_count=5, workers=10)
    return model

def doc_clustering(model, cluster_num):
    doc_num = len(model.docvecs.doctags.keys())
    train_data = np.array([model.docvecs['a_' + str(doc + 1)] for doc in range(doc_num)])
    clusterer = SphericalKMeans(cluster_num)
    print('Start clustering...')
    clusterer.fit(train_data)
    print('Done.')
    return clusterer

def get_cluster_membership(clusterer):
    # For each doc, we get which cluster it belongs to.
    cluster_membership = defaultdict(list)
    labels = clusterer.labels_
    for idx, label in enumerate(labels):
        cluster_membership[label].append(idx)
    for k in cluster_membership:
        cluster_membership[k] = ['a_' + str(vv + 1) for vv in cluster_membership[k]]
    return cluster_membership

def calculate_target_phrase_freq(document_phrase_cnt, target_doc_set):
    phrase2freq = defaultdict(int)
    for doc_id in target_doc_set:
        for phrase in document_phrase_cnt['a_' + str(doc_id)]:
            phrase2freq[phrase] += document_phrase_cnt['a_' + str(doc_id)][phrase]
    return dict(phrase2freq)

def get_doc_membership(cluster_membership, doc_id):
    for k in cluster_membership:
        if 'a_' + str(doc_id) in cluster_membership[k]:
            return k
    return -1

def rank_docs_similarity(target_set, cluster_members, model, reverse=True):
    # For target doc sets, rank the other sets in the same cluster based on cosine similarity.
    l = []
    for cluster_member in cluster_members:
        if int(cluster_member[2:]) in target_set:
            continue
        l.append((cluster_member,
                  np.mean([model.docvecs.similarity(cluster_member, 'a_' + str(doc_id)) for doc_id in target_set])))
    return sorted(l, key=lambda t: t[1], reverse=reverse)
