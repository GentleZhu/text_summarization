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
import sys,pickle
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
        print(idx, records[idx]['abstract'],records[idx]['lead_3'])
        f = collect_extractive_fragments(word_tokenize(records[idx]['abstract']), word_tokenize(records[idx]['content']))
        #print(f)
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
    #print(gt_fragments)
    return 1.0 * correct / len(gt_fragments), correct_phrases

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
    chosen_sentences = [sentences[i].replace('<phrase>', '').replace('</phrase>', '') for i in chosen]
    return chosen_sentences

def evaluate_rouge(reference, hypothesis):
    rouge = Rouge()
    return rouge.get_scores(hypothesis, reference)

def read_passage(seg_file='/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt'):
    segIn = open(seg_file)
    passages = segIn.readlines()
    return passages

if __name__ == '__main__':
    if sys.argv[1] == 'generate':
        ##########################
        # TODO: (jingjing)
        # Automatic generate mutiple test set #
        # use NYT category to generate similar documents
        ##########################
        cnt = 0
        with open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json') as IN, open('terrorism_case.txt', 'w') as OUT:
            for line in IN:
                tmp = json.loads(line)
                for t in tmp['type']:
                    if 'Top/News/U.S.' in t and 'terrorism' in ' '.join(tmp['abstract']):
                        cnt += 1
                        OUT.write(' '.join(tmp['content']) + '\n')
                        break
        #print(cnt)
    
    elif sys.argv[1] == 'eval-single':
        ##########################
        # Single doc rouge block #
        ##########################

        doc_id = 5804
        budget = 500
        passages = read_passage()
        json_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json'
        records, abstracts, contents = load_json(json_path)
        ranked_list = pickle.load(open('../models/data/ranked_list.p', 'rb'))
        phrase_scores = {t[0]:t[1] for t in ranked_list}
        sentences_choice = pick_sentences(phrase_scores, 500, passages[doc_id])
        rouge_scores = evaluate_rouge(abstracts[doc_id], ' '.join(sentences_choice))
        embed()
        exit()
    elif sys.argv[1] == 'eval-multi':
        #####################
        # Phrase eval block #
        #####################
        json_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json'
        records, abstracts, contents = load_json(json_path)
        phrases = pickle.load(open('../models/data/target_phrases.p', 'rb'))
        #gt_fragments = collect_set_fragment(records, [846, 845, 2394, 2904, 2633, 2565, 2956, 2728, 2491])
        gt_fragments = collect_set_fragment(records, [5804, 5803, 17361, 20859, 18942, 18336, 21233, 19615, 17945])
        #
        _, gt = calculate_phrase_level_precision(phrases, gt_fragments)

        ranked_list = pickle.load(open('../models/data/ranked_list.p', 'rb'))
        #ap = calculate_AP(ranked_list, gt.keys())

        dcg = calculate_DCG(ranked_list, gt)
        embed()
        exit()