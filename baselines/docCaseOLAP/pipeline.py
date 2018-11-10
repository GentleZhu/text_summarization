# Currently supports rank phrases based on tf and idf.

import json,sys,itertools
from collections import defaultdict
import argparse
#import nltk.data
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import re,math
import numpy as np
from string import punctuation
from IPython import embed
from tqdm import tqdm

from doc_clustering import *

cluster_number = 50

class Summarizer:
    def __init__(self):
        self.stopwords = set()
        self.passages = []
        self.document_phrase_cnt = defaultdict(lambda: defaultdict(int))
        self.inverted_index = defaultdict(lambda: defaultdict(int))

    def load_stopwords(self, in_file):
        with open(in_file) as IN:
            for line in IN:
                self.stopwords.add(line.strip())

    def load_corpus(self, jsonIn):
        doc_id = 0
        for jline in tqdm(jsonIn):
            doc_id += 1
            tmp = json.loads(jline)
            tmp_passage = []
            term_with_tag = list(zip(tmp['token'], tmp['pos']))
            start2entity_idx_mapping = {}
            start_indices = set()
            for ner_idx, ner in enumerate(tmp['ner']):
                start_indices.add(ner[1])
                start2entity_idx_mapping[ner[1]] = ner_idx
            idx = 0
            while idx < len(term_with_tag):
                (term, pos) = term_with_tag[idx]
                if idx in start_indices:
                    ner_idx = start2entity_idx_mapping[idx]
                    t = tmp['ner'][ner_idx][0].lower().replace(' ', '_')
                    self.document_phrase_cnt['a_' + str(doc_id)][t] += 1
                    self.inverted_index[t]['a_' + str(doc_id)] += 1
                    tmp_passage.append(t)
                    idx = tmp['ner'][ner_idx][2]
                    continue
                if len(pos) > 0 and pos[0] in ['N', 'J']:
                    t = term.lower().replace(' ', '_')
                    if t not in self.stopwords:
                        tmp_passage.append(t)
                        self.document_phrase_cnt['a_' + str(doc_id)][t] += 1
                        self.inverted_index[t]['a_' + str(doc_id)] += 1
                idx += 1
            self.passages.append(tmp_passage)
        return self.passages, self.document_phrase_cnt, self.inverted_index

    def rank_phrase_tfidf(self, passage_id, model):
        # Used to simply rank a document using tf-idf to get rough information.
        phrase_candidates = list(self.document_phrase_cnt['a_' + str(passage_id)].keys())
        ranked_list = []
        for candidate in phrase_candidates:
            if candidate not in model.wv:
                continue
            tf = self.document_phrase_cnt['a_' + str(passage_id)][candidate]
            idf = math.log(len(self.document_phrase_cnt) / len(self.inverted_index[candidate]))
            score = tf * idf
            ranked_list.append((candidate, score))
        ranked_list = sorted(ranked_list, key=lambda t: -t[1])
        return ranked_list

    def rank_phrase_tfidf_multi(self, passage_id_list, model):
        # Used to rank phrases based on tf-idf for target set.
        phrase_candidates = []
        for passage_id in passage_id_list:
            phrase_candidates.extend(list(self.document_phrase_cnt['a_' + str(passage_id)].keys()))
        ranked_list = []
        for candidate in phrase_candidates:
            if candidate not in model.wv:
                continue
            tf = sum([self.document_phrase_cnt['a_' + str(passage_id)][candidate] for passage_id in passage_id_list])
            idf = math.log(len(self.document_phrase_cnt) / len(self.inverted_index[candidate]))
            score = tf * idf
            ranked_list.append((candidate, score))
        ranked_list = sorted(ranked_list, key=lambda t: -t[1])
        return ranked_list

    def calculate_similarity(self, target_set, background_id, model, k=10):
        target_key = self.rank_phrase_tfidf_multi(target_set, model)[:k]
        background_key = self.rank_phrase_tfidf(background_id, model)[:k]
        scores = [max([model.wv.similarity(key[0], b_key[0]) for b_key in background_key]) for key in target_key]
        similarity_score = np.mean(scores)
        return similarity_score

    def embed_cluster(self):
        # Train document embedding and document clustering.
        model = train_doc2vec(self.passages)
        clusterer = doc_clustering(model, cluster_number)
        cluster_membership = get_cluster_membership(clusterer)

        return model, cluster_membership

    def cluster_tfidf_similarity(self, sorted_cluster_list, target_set, k=5):
        scores = []
        for t in sorted_cluster_list[:k]:
            score = np.mean([model.docvecs.similarity(t[0], 'a_' + str(doc_id)) for doc_id in target_set])
            scores.append(score)
        return np.mean(scores)

    def rank_cluster(self, cluster_membership, target_set, model):
        sorted_list = []
        topk = 10
        for cluster_idx in range(cluster_number):
            ranked_docs = rank_docs_similarity(target_set, cluster_membership[cluster_idx], model)[:topk]
            embedding_score = np.mean([t[1] for t in ranked_docs])
            tfidf_score = self.cluster_tfidf_similarity(ranked_docs, target_set)
            similarity_score = embedding_score * tfidf_score
            sorted_list.append((cluster_idx, similarity_score))
        sorted_list = sorted(sorted_list, key=lambda t: -t[1])
        return sorted_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarizer for text summariztion.")
    parser.add_argument("--in1", default='/shared/data/qiz3/text_summ/data/nyt13_110k_summ.json', help="Input class number", type=str)
    args = parser.parse_args()
    tmp = Summarizer()
    tmp.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
    jsonIn = open(args.in1)
    tmp.load_corpus(jsonIn)

    model, cluster_membership = tmp.embed_cluster()
    embed()
    exit()
