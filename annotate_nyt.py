import xml.dom.minidom
from collections import defaultdict
from tqdm import tqdm
import re
import math
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
#from IPython import embed
import sys
import random

class NYTAnnotator:
    def __init__(self, passage_collection):
        self.passage_collection = dict()

        for passage_data in passage_collection:
            self.passage_collection[passage_data['id']] = passage_data

        self.idf = set()
        self.document_phrase_cnt = defaultdict(lambda: defaultdict(int))
        self.inverted_index = defaultdict(lambda: defaultdict(int))
        self.stopwords = set()
        self.idf = dict()
        self.keyphrase_annotations = defaultdict(list)

        self.collect_statistics()
        self.calculate_idf()

    def collect_statistics(self):
        stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
        with open(stopword_path) as IN:
            for line in IN:
                self.stopwords.add(line.strip())
        for passage_data_idx in tqdm(self.passage_collection):
            passage_data = self.passage_collection[passage_data_idx]
            text = passage_data['text']
            id = passage_data['id']
            words = word_tokenize(text)
            for t in words:
                t = re.sub(r'\W+', '', t)
                if t.lower() not in self.stopwords:
                    _pos = wn.synsets(t)
                    if len(_pos) > 0:
                        _pos = _pos[0].pos()
                    # print(_pos)
                    if _pos == 'n' or _pos == 'a' or '_' in t:
                        self.document_phrase_cnt[id][t.lower()] += 1
                        self.inverted_index[t.lower()][id] += 1

    def dump(self, outfile):
        with open(outfile, 'w') as OUT:
            for k in self.keyphrase_annotations:
                OUT.write("{}\n".format(' '.join(self.keyphrase_annotations[k])))


    def calculate_idf(self):
        assert(len(self.document_phrase_cnt) > 0)
        total_num = len(self.passage_collection)
        for candidate in self.inverted_index:
            p_freq = len(self.inverted_index[candidate])
            self.idf[candidate] = math.log(total_num / p_freq)
        print('Idf calculation done.')

    def display_passage(self, abstract, text, candidates):
        print('\033[1;31;46mAbstract: \033[0m')
        print(abstract)
        print('\033[1;31;46mText: \033[0m')
        print(text)
        print('\033[1;31;46mTF-IDF candidates: \033[0m')
        print(candidates)
        print('\033[1;31;43mChoose one option: \nq: quit; h: highlight; s: summarize\033[0m')

    def display(self, max_id):
        for id in range(max_id):
            passage_data = self.passage_collection[id]
            sorted_list = []
            for candidate in self.document_phrase_cnt[id]:
                sorted_list.append((candidate, self.document_phrase_cnt[id][candidate] * self.idf[candidate]))
            sorted_list = sorted(sorted_list, key=lambda t: -t[1])
            candidates = [t[0] for t in sorted_list[:20]]
            random.shuffle(candidates)
            self.display_passage(passage_data['abstract'], passage_data['text'], candidates)
            while True:
                choice = input('Type your option: ')
                if choice == 'q':
                    break
                elif choice == 's':
                    keyphrases = input('Please input key phrases, divided by semi-colon: ')
                    keyphrases = [x.strip() for x in keyphrases.split(';')]
                    self.keyphrase_annotations[id] = keyphrases
                    break
                elif choice == 'h':
                    target = input('Please input your interested phrase: ').strip()
                    new_text = passage_data['text'].replace(target, '\033[1;31;46m' + target + '\033[0m')
                    self.display_passage(passage_data['abstract'], new_text, candidates)
                else:
                    print('Invalid input!')

def rank_tf(text):
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    stopwords = set()
    with open(stopword_path) as IN:
        for line in IN:
            stopwords.add(line.strip())
    words = word_tokenize(text)
    bow = defaultdict(int)
    for t in words:
        t = re.sub(r'\W+', '', t)
        if t.lower() not in stopwords:
            _pos = wn.synsets(t)
            if len(_pos) > 0:
                _pos = _pos[0].pos()
            # print(_pos)
            if _pos == 'n' or _pos == 'a' or '_' in t:
                bow[t.lower()] += 1
    sorted_list = sorted([(k, bow[k]) for k in bow], key=lambda t: -t[1])
    return sorted_list

if __name__ == '__main__':
    ct = 0
    passage_collection = []
    with open("/shared/data/yuningm2/datasets/NYT_annotated_corpus/data/filelist_2003-07.txt") as f:
        for line in tqdm(f):
            xml_path = "/shared/data/yuningm2/datasets/NYT_annotated_corpus/data/accum2003-07/" + line.strip()
            passage_data = {}
            #with open(xml_path) as f2:
                #print(f2.readlines())
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement

            tags = root.getElementsByTagName('abstract')
            if len(tags) > 0:
                abstract = ''
                for tag in tags:
                    data = tag.childNodes[1].childNodes[0].data
                    abstract += data
                passage_data['abstract'] = abstract

                tags = root.getElementsByTagName('title')
                for tag in tags:
                    data = tag.firstChild.data
                    passage_data['title'] = data

                full_text = ''
                tags = root.getElementsByTagName('block')
                for tag in tags:
                    bclass = tag.getAttribute('class')
                    # remove lead_paragraph
                    if bclass == 'full_text':
                        ps = tag.getElementsByTagName('p')
                        for p in ps:
                            data = p.firstChild.data
                            full_text += data
                passage_data['text'] = full_text
                if len(passage_data['text']) > 2000:
                    passage_data['id'] = ct
                    ct += 1
                    passage_collection.append(passage_data)
            if ct > int(sys.argv[1]):
                break

    annotator = NYTAnnotator(passage_collection)
    annotator.display(int(sys.argv[1]))
    annotator.dump('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/'+sys.argv[2]+'.txt')
    #embed()