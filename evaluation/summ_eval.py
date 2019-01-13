from IPython import embed
import nltk
import re
from math import log
import json
import itertools
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import math
import pickle

from rouge import Rouge

def load_json(json_path):
    records = []
    abstracts = []
    contents = []
    f = open(json_path, 'r')
    for line in tqdm(f):
        record = json.loads(line)
        abstract = ''
        content = ''
        for sentence in record['abstract']:
            abstract += sentence
        for sentence in record['content']:
            content += sentence
        record['abstract'] = abstract
        record['content'] = content
        records.append(record)
        abstracts.append(abstract)
        contents.append(content)
    return records, abstracts, contents

def convert_line_number(old_ids):
    old_file = '/shared/data/qiz3/text_summ/data/NYT_sports.token'
    new_file = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.txt'
    new_ids = []
    with open(old_file, 'r') as OLD, open(new_file, 'r') as NEW:
        old_lines = OLD.readlines()
        new_lines = NEW.readlines()
        new_content = [new_lines[x].replace(' ', '')[:100] for x in tqdm(range(len(new_lines)))]
        for id in old_ids:
            flag = False
            content = old_lines[id].replace('_', ' ').replace(' ', '')[:100]
            for idx, x in enumerate(new_content):
                if x == content:
                    new_ids.append(idx)
                    flag = True
                    break
            if not flag:
                new_ids.append(-1)
    return new_ids

def collect_extractive_fragments(abstract, content):
    tagged = nltk.pos_tag(abstract)
    extractive_fragments = []
    i, j = (0, 0)
    while i < len(abstract):
        if tagged[i][1] == 'IN':
            i += 1
            continue
        f = []
        while j < len(content):
            if abstract[i] == content[j] and not abstract[i] in [',', '.', '?', '!', ':']:
                i1, j1 = i, j
                while i1 < len(abstract) and j1 < len(content) and abstract[i1] == content[j1] and not abstract[i1] in [',', '.', '?', '!', ':']:
                    i1 += 1
                    j1 += 1
                if len(f) < i1 - i:
                    f = abstract[i:i1]
                j = j1
            else:
                j += 1
        i, j = (i + max(len(f), 1), 0)
        if len(f) > 1:
            extractive_fragments.append(f)
    return extractive_fragments

def calculate_coverage(abstract, content):
    extractive_f = collect_extractive_fragments(abstract, content)
    return sum([len(f) for f in extractive_f]) / (len(abstract) + 0.01)

def calculate_density(abstract, content):
    extractive_f = collect_extractive_fragments(abstract, content)
    return sum([len(f) * len(f) for f in extractive_f]) / (len(abstract) + 0.01)

def calculate_compression(abstract, content):
    if len(abstract) == 0:
        return 0
    return len(content) / len(abstract)

def collect_set_fragment(records, passage_ids):
    fragments = []
    for idx in passage_ids:
        f = collect_extractive_fragments(word_tokenize(records[idx]['abstract']), word_tokenize(records[idx]['content']))
        fragments.extend(f)
    return fragments

def calculate_phrase_level_precision(phrases, gt_fragments):
    total_n = len(phrases)
    correct = 0
    correct_phrases = defaultdict(int)
    for phrase in phrases:
        for fragment in gt_fragments:
            if phrase.lower() in '_'.join(fragment).lower():
                if phrase not in correct_phrases:
                    correct += 1
                correct_phrases[phrase] += 1
    #ranked_list = [(k, correct_phrases[k]) for k in correct_phrases]
    #ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    return 1.0 * correct / total_n, correct_phrases

def calculate_AP(ranked_list, ground_truth):
    # calculate average precision (area under PRC)
    gt = []
    predict = []
    for phrase, score in ranked_list:
        if phrase in ground_truth:
            gt.append(1)
        else:
            gt.append(0)
        predict.append(score)
    y_true = np.array(gt)
    y_scores = np.array(predict)
    return average_precision_score(y_true, y_scores)

def calculate_DCG(ranked_list, ground_truth, normalized=False):
    score = 0.
    for idx, (phrase, _) in enumerate(ranked_list):
        if normalized and idx > len(ground_truth):
            break
        if not phrase in ground_truth:
            continue
        score += ground_truth[phrase] * 1.0 / math.log(2 + idx)
    return score

def evaluate_rouge(reference, hypothesis):
    rouge = Rouge()
    return rouge.get_scores(hypothesis, reference)

if __name__ == '__main__':
    json_path = '/shared/data/qiz3/text_summ/NYT_sports.json'
    records, abstracts, contents = load_json(json_path)
    phrases = pickle.load(open('../models/data/target_phrases.p', 'rb'))
    gt_fragments = collect_set_fragment(records, [846, 845, 2394, 2904, 2633, 2565, 2956, 2728, 2491])
    _, gt = calculate_phrase_level_precision(phrases, gt_fragments)

    ranked_list = pickle.load(open('../models/data/ranked_list.p', 'rb'))
    #ap = calculate_AP(ranked_list, gt.keys())

    dcg = calculate_DCG(ranked_list, gt)

    embed()
    exit()
