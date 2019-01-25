from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from IPython import embed
from tqdm import tqdm
from glob import glob
import os
import subprocess
from nltk.stem.porter import *

duc_set = ['d30001t', 'd30002t', 'd30003t', 'd30005t', 'd30006t', 'd30007t', 'd30008t', 'd30010t', 'd30011t',
           'd30015t', 'd30017t', 'd30020t', 'd30022t', 'd30024t', 'd30026t', 'd30027t', 'd30028t', 'd30029t',
           'd30031t', 'd30033t', 'd30034t', 'd30036t', 'd30037t', 'd30038t', 'd30040t', 'd30042t', 'd30044t',
           'd30045t', 'd30046t', 'd30047t']

def load_corpus(corpusIn, target_set, stopword_path):
    stemmer = PorterStemmer()
    doc_id = -1
    passages = []
    raw_sentences = []
    stopwords = set()
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())
    for cline in tqdm(corpusIn):
        doc_id += 1
        if not doc_id in target_set:
            continue
        sentences = sent_tokenize(cline)
        for sentence in sentences:
            if len(sentence) < 5:
                continue
            e_s = []
            words = word_tokenize(sentence)
            for word in words:
                if not word in stopwords and is_eng(word):
                    e_s.append(stemmer.stem(word.lower()))
            if len(e_s) > 0:
                raw_sentences.append(sentence)
                passages.append(e_s)
    return passages, raw_sentences

def generate_duc_docs(file_name, stopword_path):
    prefix = '/shared/data/qiz3/text_summ/data/TextSummarizer-master/DUC-2004/Cluster_of_Docs/'
    cwd = os.getcwd()

    stopwords = set()
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())

    stemmer = PorterStemmer()
    passages = []
    raw_sentences = []
    os.chdir(prefix)
    os.chdir(file_name)
    docs = glob('*')
    for doc in docs:
        with open(doc) as IN:
            content = IN.read()
            sentences = sent_tokenize(content)
            for sentence in sentences:
                if len(sentence) < 5:
                    continue
                e_s = []
                words = word_tokenize(sentence)
                for word in words:
                    if not word in stopwords and is_eng(word):
                        e_s.append(stemmer.stem(word.lower()))
                if len(e_s) > 0:
                    raw_sentences.append(sentence)
                    passages.append(e_s)
    os.chdir(cwd)
    return passages, raw_sentences

def calculate_len(passages, raw_sentences):
    el = {}
    rl = {}
    for idx, (passage, raw_sent) in enumerate(zip(passages, raw_sentences)):
        el[idx] = len(set(passage))
        rl[idx] = len(passage)
    return el, rl

def construct_vocab(passages):
    word2idx = {}
    for sentence in passages:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx

def construct_sentence_vec(passages):
    vocab = construct_vocab(passages)
    sent_vec = {}
    for idx, sent in enumerate(passages):
        vec = [0 for _ in vocab]
        for word in sent:
            vec[vocab[word]] = 1.0
        sent_vec[idx] = vec
    return sent_vec

def is_eng(word):
    return word[0] < 'Z' and word[0] > 'A' or word[0] > 'a' and word[0] < 'z'

def eval_duc_full():
    cwd = os.getcwd()
    os.chdir('/shared/data/qiz3/text_summ/data/TextSummarizer-master/C_Rouge')
    for s in tqdm(duc_set):
        cmd_list = ''
        cmd_list += 'java -cp C_ROUGE.jar'
        cmd_list += ' executiverouge.C_ROUGE'
        cmd_list += ' /shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/res/' + s + '.txt'
        cmd_list += ' /shared/data/qiz3/text_summ/data/TextSummarizer-master/DUC-2004/Test_Summaries/' + s
        cmd_list += ' 1 A F >> test.txt'
        os.system(cmd_list)
    os.chdir(cwd)
    IN = open('/shared/data/qiz3/text_summ/data/TextSummarizer-master/C_Rouge/test.txt')
    score = 0.0
    c = 0
    for idx, line in enumerate(IN.readlines()):
        s = float(line.strip('\n'))
        c += 1
        score += s
    return score / c

if __name__ == '__main__':
    print(eval_duc_full())
    exit()
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    corpusIn = open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.txt')
    target_set = [5804, 5803, 17361, 20859, 18942, 18336, 21233, 19615, 17945]
    passages, raw_sentences = load_corpus(corpusIn, target_set, stopword_path)
    embed()
    exit()