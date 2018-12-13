import pickle
import json,sys,itertools
from collections import defaultdict
import argparse
import nltk.data
from nltk.corpus import wordnet as wn
import re
import json
from string import punctuation
from tqdm import tqdm
import numpy as np
import math
from IPython import embed

from WikidataLinker_mg import WikidataLinker


class HinLoader(object):
    """docstring for HinLoader"""

    def __init__(self, arg, weighted=False):
        self.weighted = weighted
        if weighted:
            self.weight = list()
        self.in_mapping = dict()
        self.out_mapping = dict()
        self.input = list()
        self.output = list()
        self.arg = arg
        self.edge_stat = [0] * len(self.arg['edge_types'])
        # print(arg['types'])
        for k in arg['types']:
            self.in_mapping[k] = dict()
            self.out_mapping[k] = dict()
            # print(self.in_mapping.keys())
            # print(self.out_mapping.keys())

    def inNodeMapping(self, key, type):
        if key not in self.in_mapping[type]:
            self.out_mapping[type][len(self.in_mapping[type])] = key
            self.in_mapping[type][key] = len(self.in_mapping[type])

        return self.in_mapping[type][key]

    def readHin(self, _edge_types):
        # num_nodes = defaultdict(int)

        with open(self.arg['graph']) as INPUT:
            for line in INPUT:
                # replace original HEER with current setting
                # edge = line.strip().split(' ')
                edge = line.strip().split('\t')
                edge_type = _edge_types.index(edge[-1])
                node_a = edge[0].split(':')
                node_b = edge[1].split(':')
                node_a_type = self.arg['types'].index(node_a[0])
                node_b_type = self.arg['types'].index(node_b[0])
                # assert edge_type != 11
                self.edge_stat[edge_type] += 1
                assert [node_a_type, node_b_type] == self.arg['edge_types'][edge_type][:2]
                self.input.append([edge_type, self.inNodeMapping(node_a[1], node_a[0])])
                self.output.append([self.arg['types'].index(node_b[0]), self.inNodeMapping(node_b[1], node_b[0])])
                self.weight.append(float(edge[2]))

    def encode(self):
        self.encoder = dict()
        offset = 0
        for k in self.arg['types']:
            self.encoder[k] = offset
            offset += len(self.in_mapping[k])
        self.encoder['sum'] = offset
        print(self.encoder)
        for i, ie in enumerate(self.input):
            self.input[i][1] += self.encoder[self.arg['types'][self.arg['edge_types'][ie[0]][0]]]
        for i, ie in enumerate(self.output):
            self.output[i][1] += self.encoder[self.arg['types'][ie[0]]]

    def dump(self, dump_path):
        print(self.edge_stat)
        pickle.dump(self.encoder, open(dump_path + '_offset.p', 'wb'))
        pickle.dump(self.input, open(dump_path + '_input.p', 'wb'))
        pickle.dump(self.output, open(dump_path + '_output.p', 'wb'))
        pickle.dump(self.in_mapping, open(dump_path + '_in_mapping.p', 'wb'))
        pickle.dump(self.out_mapping, open(dump_path + '_out_mapping.p', 'wb'))
        pickle.dump(self.edge_stat, open(dump_path + '_edge_stat.p', 'wb'))
        if self.weighted:
            pickle.dump(self.weight, open(dump_path + '_weight.p', 'wb'))

class HinBuilder:
    def __init__(self):
        self.n_doc = 0
        self.stopwords = set()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.dp_edges = defaultdict(lambda: defaultdict(int))
        self.pd_edges = defaultdict(lambda: defaultdict(int))
        self.labels = ['football', 'basketball', 'baseball', 'hockey', 'golf', 'tennis']#set()

    def load_stopwords(self, in_file='/shared/data/qiz3/text_summ/data/stopwords.txt'):
        with open(in_file) as IN:
            for line in IN:
                self.stopwords.add(line.strip().lower())

    def build_graph(self, jsonIn=open('/shared/data/qiz3/text_summ/data/NYT_sports.json')):
        self.load_stopwords()
        doc_id = 0
        print('Processing text...')
        for jline in tqdm(jsonIn):
            doc_id += 1
            #if doc_id > 1000:
            #    break
            tmp = json.loads(jline)
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
                    if t != '\n':
                        t = t.replace('\n', '')
                        self.dp_edges[doc_id-1][t] += 1
                        self.pd_edges[t][doc_id-1] += 1
                        idx = tmp['ner'][ner_idx][2]
                        continue
                if len(pos) > 0 and pos[0] in ['N', 'J']:
                    t = term.lower().replace(' ', '_')
                    if t not in self.stopwords:
                        if t != '\n':
                            t = t.replace('\n', '')
                            self.dp_edges[doc_id-1][t] += 1
                            self.pd_edges[t][doc_id-1] += 1
                idx += 1

        self.indexing_edges(doc_id)

    def build_graph_autophrase(self, in_file='/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/news.txt'):
        n_doc = 0
        with open(in_file) as IN:
            for line in IN:
                n_doc += 1
                line = line.strip('\n').split('\t')
                if line[1] != '':
                    doc_id, phrases = int(line[0]), line[1].split(';')
                else:
                    continue
                for t in phrases:
                    self.dp_edges[doc_id][t] += 1
                    self.pd_edges[t][doc_id] += 1

        self.indexing_edges(n_doc)

    def extract_labels(self, jsonIn = open('/shared/data/qiz3/text_summ/data/NYT_sports.json'), relation_list=['P31']):
        self.labels = []
        linker = WikidataLinker(relation_list)
        doc_id = 0
        print('Extracting labels...')
        for jline in tqdm(jsonIn.readlines()):
            doc_id += 1
            if doc_id > 10:
                break
            ner = json.loads(jline)['ner']
            d = [t for t in linker.expand(ner, 1) if 'disam' not in t[1]]
            #d = [t[0] for t in d]
            self.labels.append(ner)

    def extract_typed_ner(self, jsonIn = open('/shared/data/qiz3/text_summ/data/NYT_sports.json'), type='LOC'):
        for jline in tqdm(jsonIn.readlines()):
            ner = json.loads(jline)['ner']
            for n in ner:
                if n[3] == type:
                    t = n[0].lower().replace(' ', '_')
                    self.labels[t.replace('\n', '')] += 1
        self.labels = set(sorted(list(self.labels.keys()), key=lambda t: self.labels[t])[:10])

    def generate_static_location_label(self):
        self.labels = set(['california', 'florida', 'texas', 'hawaii', 'arizona', 'washington', 'north_carolina', 'pennsylvania',
                           'georgia', 'new_york', 'michigan', 'new_jersey', 'virginia', 'massachusetts', 'colorado',
                           'ohio', 'illinois', 'alaska', 'minnesota', 'alabama', 'tennessee', 'missouri', 'maryland',
                           'oregon', 'south_carolina', 'indiana', 'wisconsin', 'utah', 'louisiana', 'connecticut',
                           'kentucky', 'maine', 'oklahoma', 'iowa', 'nevada', 'montana', 'mississippi', 'kansas',
                           'arkansas', 'nebraska', 'wyoming', 'rhode_island', 'west_virginia', 'delaware',
                           'vermont', 'new_hampshire', 'idaho', 'south_dakota', 'north_dakota', 'new_mexico'])

    def generate_static_topic_label(self):
        self.labels = set(['federal_budget', 'surveillance', 'dance', 'immigration',
                           'cosmos', 'television', 'law_enforcement', 'hockey', 'business',
                           'basketball', 'gay_rights', 'science', 'tennis', 'golf', 'football', 'music',
                           'economy', 'gun_control', 'baseball', 'soccer', 'politics', 'military', 'abortion'])

    def indexing_edges(self, doc_id):
        self.doc_nodes = list(range(doc_id))
        self.phrase_nodes = list(self.pd_edges.keys())

    def dump_edges(self, doc_node_file='../doc2cube/tmp_jt/doc_node.txt', phrase_node_file='../doc2cube/tmp_jt/phrase_node.txt',
                   label_node_file='../doc2cube/tmp_jt/label_node.txt', edge_file='../doc2cube/tmp_jt/edge.txt'):
        with open(doc_node_file, 'w') as D_OUT:
            for doc_idx, doc_id in enumerate(self.doc_nodes):
                D_OUT.write(str(doc_idx) + '\t' + str(doc_id) + '\n')

        with open(phrase_node_file, 'wb') as P_OUT:
            for phrase_idx, phrase in enumerate(self.phrase_nodes):
                P_OUT.write((str(phrase_idx) + '\t' + phrase + '\n').encode('utf-8'))

        with open(label_node_file, 'wb') as L_OUT:
            for l_idx, label in enumerate(self.labels):
                L_OUT.write((str(l_idx) + '\t' + label + '\n').encode('utf-8'))

        phrase_idx = {phrase: idx for idx, phrase in enumerate(self.phrase_nodes)}

        with open(edge_file, 'w') as E_OUT:
            print('Dumping edges...')
            for doc_idx, doc_id in tqdm(enumerate(self.doc_nodes)):
                for phrase in self.dp_edges[doc_id]:
                    E_OUT.write('D:' + str(doc_idx) + '\t' + 'P:' + str(phrase_idx[phrase]) + '\t' +
                                str(self.dp_edges[doc_id][phrase]) + '\tDP' + '\n')

            #return

            for label_idx, label in enumerate(self.labels):
                if label in phrase_idx:
                    p_idx = phrase_idx[label]
                    E_OUT.write('L:' + str(label_idx) + '\t' + 'P:' + str(p_idx) + '\t1\tLP' + '\n')
                else:
                    print(label)


if __name__ == '__main__':
    jsonIn = open('/shared/data/qiz3/text_summ/data/NYT_sports.json')
    d = defaultdict(int)
    for jline in tqdm(jsonIn.readlines()):
        ner = json.loads(jline)['ner']
        for n in ner:
            if n[3] == 'PERSON':
                t = n[0].lower().replace(' ', '_')
                d[t] += 1
    ranked_list = [(t, d[t]) for t in d]
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    embed()
    exit()

    builder = HinBuilder()
    builder.build_graph_autophrase()
    builder.extract_labels()
    #builder.extract_typed_ner()
    #builder.generate_static_location_label()
    #builder.generate_static_topic_label()
    #builder.dump_edges()
    embed()
    exit()
    builder.extract_labels()
