from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from IPython import embed
from tqdm import tqdm
from glob import glob
import os
import subprocess
import pickle
from nltk.stem.porter import *

#duc_set = ['d30048t', 'd30049t', 'd30050t', 'd30051t', 'd30053t', 'd30055t', 'd30056t', 'd30059t', 'd31001t',
#           'd31008t', 'd31009t', 'd31013t', 'd31022t', 'd31026t', 'd31031t', 'd31032t', 'd31033t', 'd31038t',
#           'd31043t', 'd31050t']
duc_set =  ['2018_ca_wildfire', 'Indiahomo_201809', 'Mars_201807', 'Roaster_201802','Tsunami_201812', 'Bridge_201810', 'James_201808', 'Rhino_201803', 'Tigerwoods_201809', 'final_debate']
#duc_set = ['d30001', 'd30002', 'd30003', 'd30005','d30006', 'd30007', 'd30015', 'd30033', 'd30034', 'd30036']

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
    prefix = '/shared/data/qiz3/text_summ/text_summarization/results/RPN/'

    stopwords = set()
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())

    passages = []
    raw_sentences = []
    for idx, doc in enumerate(open(prefix + file_name + '.in')):
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
    for idx, sentence in enumerate(open(file_path).readlines()):
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

        passages.append(word_tokenize(sentence))
    return passages, raw_sentences

def calculate_len(passages, raw_sentences):
    el = {}
    rl = {}
    for idx, (passage, raw_sent) in enumerate(zip(passages, raw_sentences)):
        words = word_tokenize(raw_sentences[idx])
        el[idx] = len(set(passage))
        rl[idx] = len(words)
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
    r.system_dir = '/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/tmp/system1/'
    r.model_dir = '/shared/data/qiz3/text_summ/text_summarization/baselines/sentence_summ/tmp/model1/'
    r.system_filename_pattern = 'd(\d+)t.txt'
    r.model_filename_pattern = 'D#ID#.M.100.T.[A-Z]'

    output = r.convert_and_evaluate()
    print(output)

def process():
    #d_l = ['d30001.txt', 'd30007.txt', 'd30017.txt', 'd30027.txt', 'd30036.txt', 'd30044.txt', 'd30002.txt',
    #       'd30008.txt', 'd30020.txt', 'd30028.txt', 'd30037.txt', 'd30046.txt', 'd30003.txt', 'd30010.txt',
    #       'd30022.txt', 'd30029.txt', 'd30038.txt', 'd30047.txt', 'd30005.txt', 'd30011.txt', 'd30024.txt',
    #       'd30033.txt', 'd30040.txt', 'd30006.txt', 'd30015.txt', 'd30026.txt', 'd30034.txt', 'd30042.txt']
    d_l = ['2018_ca_wildfire.in', 'James_201808.in', 'Tigerwoods_201809.in', 'Bridge_201810.in', 'Mars_201807.in',
           'Tsunami_201812.in', 'final_debate.in', 'Rhino_201803.in', 'Indiahomo_201809.in', 'Roaster_201802.in']
    for d in d_l:
        source = '/shared/data/qiz3/text_summ/text_summarization/data/news_doc_line/' + d
        target = '/shared/data/qiz3/text_summ/text_summarization/data/news_doc_line/' + d
        IN = open(source)
        content = IN.readlines()
        IN.close()
        OUT = open(target, 'w')
        for line in content:
            sentences = sent_tokenize(line)
            for sentence in sentences:
                sentence = sentence.strip('\n')
                OUT.write(sentence + '\n')

if __name__ == '__main__':
    process()
    exit()
    evaluate_rouge155()