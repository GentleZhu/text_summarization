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

    def rank_phrase_tfidf(self, passage_id):
        # Used to simply rank a document using tf-idf to get rough information.
        phrase_candidates = list(self.document_phrase_cnt[passage_id].keys())
        ranked_list = []
        for candidate in phrase_candidates:
            tf = self.document_phrase_cnt[passage_id][candidate]
            idf = math.log(len(self.document_phrase_cnt) / len(self.inverted_index[candidate]))
            score = tf * idf
            ranked_list.append((candidate, score))
        ranked_list = sorted(ranked_list, key=lambda t: -t[1])
        return ranked_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarizer for text summariztion.")
    parser.add_argument("--in1", default='/shared/data/qiz3/text_summ/data/nyt13_110k_summ.json', help="Input class number", type=str)
    args = parser.parse_args()
    tmp = Summarizer()
    tmp.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
    with open(args.in1) as jsonIn:
        passages, document_phrase_cnt, inverted_index = tmp.load_corpus(jsonIn)
        embed()
        exit()
