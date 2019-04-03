from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from IPython import embed
from tqdm import tqdm
from glob import glob
import os
import subprocess
from nltk.stem.porter import *

#duc_set = ['d30048t', 'd30049t', 'd30050t', 'd30051t', 'd30053t', 'd30055t', 'd30056t', 'd30059t', 'd31001t',
#           'd31008t', 'd31009t', 'd31013t', 'd31022t', 'd31026t', 'd31031t', 'd31032t', 'd31033t', 'd31038t',
#           'd31043t', 'd31050t']
duc_set = ['d30002']

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
    prefix = '/shared/data/qiz3/text_summ/data/DUC04/train/'

    stopwords = set()
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())

    passages = []
    raw_sentences = []
    for idx, doc in enumerate(open(prefix + file_name + '.txt')):
        content = doc.strip('\n')
        sentences = sent_tokenize(content)
        for sentence in sentences:
            e_s = []
            raw_sentences.append(sentence)
            words = word_tokenize(sentence)
            for word in words:
                if not word in stopwords:
                    e_s.append(word.lower())
            passages.append(e_s)
    return passages, raw_sentences

def generate_duc_docs_autophrase(file_name):
    file_path = '/shared/data/qiz3/text_summ/src/jt_code/HiExpan/data/nyt/intermediate/segmentation1.txt'

    index = duc_set.index(file_name)
    r_l = 167842 + index * 10
    r_r = r_l + 10
    passages = []
    raw_sentences = []
    for idx, content in enumerate(open(file_path).readlines()):
        if idx < r_l or idx >= r_r:
            continue
        content = content.strip('\n')
        sentences = sent_tokenize(content)
        for sentence in sentences:
            if len(sentence) < 5:
                continue
            phrases = re.findall('<phrase>.*?</phrase>', sentence)
            phrases = [phrase[8:-9].replace(' ', '_').lower() for phrase in phrases]
            if len(phrases) > 0:
                raw_sentences.append(sentence.replace('<phrase>', '').replace('</phrase>', ''))
                passages.append(phrases)
    return passages, raw_sentences

def generate_docs_autophrase(file_path):
    passages = []
    raw_sentences = []
    for idx, content in enumerate(open(file_path).readlines()):
        content = content.strip('\n')
        sentences = sent_tokenize(content)
        for sentence in sentences:
            raw_sentences.append(sentence.replace("<phrase>", "").replace("</phrase>",""))
            sentence = sentence.lower()
            #if 'malibu' in sentence or 'woolsey' in sentence:
            #    print(sentence)
            #if len(sentence) < 5:
            #    continue
            phrases = re.findall('<phrase>(.*?)</phrase>', sentence)
            #phrases = [phrase[8:-9].replace(' ', '_').lower() for phrase in phrases]
            #print(phrases)
            for p in set(phrases):
                sentence = sentence.replace("<phrase>"+p+"</phrase>", p.replace(' ', '_'))
            #if len(phrases) > 0:
            
            passages.append(sentence.split())
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
    if os.path.exists('test.txt'):
        os.system('rm test.txt')
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

def evaluate_rouge155():
    from pyrouge import Rouge155

    r = Rouge155()
    r.system_dir = '/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/tmp/system/'
    r.model_dir = '/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/tmp/model/'
    r.system_filename_pattern = 'd(\d+)t.txt'
    r.model_filename_pattern = 'D#ID#.M.100.T.[A-Z]'

    output = r.convert_and_evaluate()
    print(output)

if __name__ == '__main__':
    evaluate_rouge155()
    exit()
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    corpusIn = open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.txt')
    target_set = [5804, 5803, 17361, 20859, 18942, 18336, 21233, 19615, 17945]
    passages, raw_sentences = load_corpus(corpusIn, target_set, stopword_path)
    embed()
    exit()