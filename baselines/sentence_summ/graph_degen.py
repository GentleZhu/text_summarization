import glob
import os
import re
import math
import numpy as np
import nltk
import pickle
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from IPython import embed
from tqdm import tqdm

from utils import *

class Node_info:
    def __init__(self, name):
        self.name = name
        self.weight = 0
        self.core_n = 0

class Graph_info:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.name2idx  = {}
        for idx, node in enumerate(self.nodes):
            self.name2idx[node.name] = idx

def from_terms_to_graph(sentences, window_size):
    unique_terms = set()
    nodes = []
    edges = defaultdict(lambda: defaultdict(int))
    long_sentence = []

    for sentence in sentences:
        long_sentence.extend(sentence)
    for idx in range(len(long_sentence)):
        for w in range(window_size + 1):
            if idx + w > len(long_sentence) - 1:
                break
            term1, term2 = long_sentence[idx], long_sentence[idx + w]
            #(term1, tag1), (term2, tag2) = long_sentence[idx], long_sentence[idx + w]
            #if tag1 not in ['N', 'J'] or tag2 not in ['N', 'J']:
            #    break
            if term2 not in unique_terms:
                unique_terms.add(term2)
                nodes.append(Node_info(term2))
            if term1 == term2:
                continue
            #edges[term1][term2] += 1
            #edges[term2][term1] += 1
            edges[term1][term2] = 1
            edges[term2][term1] = 1
    graph = Graph_info(nodes, edges)
    return graph

def cores_dec(graph):

    def heapify(heap_g):
        total_num = len(heap_g)
        start = int(total_num / 2 - 1)
        while start >= 0:
            root = start
            end_ = total_num - 1
            while root * 2 + 1 <= end_:
                child = root * 2 + 1
                if child + 1 <= end_ and graph.nodes[heap_g[child]].weight > graph.nodes[heap_g[child + 1]].weight:
                    child += 1
                if graph.nodes[heap_g[root]].weight > graph.nodes[heap_g[child]].weight:
                    tmp = heap_g[root]
                    heap_g[root] = heap_g[child]
                    heap_g[child] = tmp
                    root = child
                else:
                    break
            start -= 1

    for node in graph.nodes:
        node.core_n = 0
        node.weight = sum(graph.edges[node.name].values())
    heap_list = [i for i in range(len(graph.nodes))]
    heapify(heap_list)
    deleted_vertices = set()

    while len(heap_list) > 0:
        top = heap_list[0]
        name_top = graph.nodes[top].name
        neighbors_top = sorted(graph.edges[name_top].keys())
        graph.nodes[top].core_n = graph.nodes[top].weight

        deleted_vertices.add(top)
        heap_list[0] = heap_list[-1]
        heap_list.pop(-1)

        if len(neighbors_top) > 0:
            for i in range(len(neighbors_top)):
                neighbor_name = neighbors_top[i]
                neighbor_idx = graph.name2idx[neighbor_name]
                graph.nodes[neighbor_idx].weight -= graph.edges[neighbor_name][name_top]
                graph.nodes[neighbor_idx].weight = max(graph.nodes[neighbor_idx].weight, graph.nodes[top].core_n)
        heapify(heap_list)

def calculate_subgraph_density(graph, node_names):
    v_count = len(node_names)
    if v_count < 2:
        return 0
    e_count = 0
    for i, node1 in enumerate(node_names):
        for j, node2 in enumerate(node_names):
            if i <= j:
                break
            if graph.edges[node1][node2] > 0:
                e_count += 1
    return e_count / (v_count * (v_count - 1))

def best_level_density(graph):
    node_names = [node.name for node in graph.nodes]
    core_or_truss_number = [node.core_n for node in graph.nodes]
    return core_or_truss_number, node_names

def load_target_docs(file_path, target_docs):
    segIn = open(file_path)
    doc_id = -1
    passage = []
    IN_PHRASE_FLAG = False
    for cline in tqdm(segIn):
        doc_id += 1
        if doc_id not in target_docs:
            continue
        cline = cline.replace('<phrase>', ' <phrase> ').replace('</phrase>', ' </phrase> ').replace('\n', '')
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
        passage.append(tmp_passage)
    return passage

def pick_sentences(word_scores, budget, passages, raw_sentences):
    lambda_ = 40
    c = 0.1
    sent_scores = {}
    sent_words = defaultdict(set)
    for idx, sentence in enumerate(passages):
        sent_words[idx] = set(sentence)
        score = sum([word_scores[word] for word in set(sentence)])
        sent_scores[idx] = score

    chosen = list()
    current_len = 0
    current_p = set()
    while len(chosen) < budget:
        max_ = -1
        max_idx = -1
        for idx in range(len(passages)):
            if idx in chosen:
                continue
            score_gain = sent_scores[idx] + lambda_ * len(sent_words[idx] - current_p)
            score_gain /= math.pow(len(raw_sentences[idx]), c)
            if score_gain > max_:
                max_ = score_gain
                max_idx = idx
        if max_idx != -1:
            chosen.append(max_idx)
            current_len += len(raw_sentences[max_idx])
            current_p |= sent_words[max_idx]
        else:
            break
    return chosen

def main():
    window = 4
    ap = False
    if ap:
        budget = 1000
    else:
        budget = 3
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    ret = {}
    for ii, s in enumerate(duc_set):
        print('Running graph_degen for %d doc...' % ii)
        passages, raw_sentences = generate_duc_docs(s, stopword_path)
        graph = from_terms_to_graph(passages, window)
        cores_dec(graph)
        core_n, names = best_level_density(graph)
        d = {name: n for name, n in zip(names, core_n)}
        word_scores = {}
        for name in names:
            score = 0
            for neighbor in graph.edges[name]:
                score += d[neighbor]
            word_scores[name] = score
        chosen = pick_sentences(word_scores, budget, passages, raw_sentences)

        #summary = ''
        #for id in chosen:
        #    summary += raw_sentences[id] + ' '

        l = [-1 for _ in passages]
        if ap:
            for idx, i in enumerate(chosen):
                l[i] = len(chosen) - idx
            ret[s] = l
        else:
            for i in chosen:
                l[i] = 1
            ret[s] = l

        #f = open('/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/tmp/system1/' + s + 't.txt',
        #         'w')
        #f.write(summary)
        #f.close()

    pickle.dump(ret, open('./data/graph_degen_sentence_res.p', 'wb'))

if __name__ == '__main__':
    main()
