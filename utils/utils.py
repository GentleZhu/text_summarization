import string
import os
import json
import io
import tarfile
import collections
import numpy as np
from tqdm import tqdm
from IPython import embed
from collections import defaultdict
from copy import deepcopy
import pickle
import operator
import math

class textGraph(object):
    """docstring for textGraph"""
    def __init__(self, arg=None):
        super(textGraph, self).__init__()
        self.name2id = dict()
        self.id2name = dict()
        self.stopwords = set()
        self.Linker = arg
        self.tuples = []
        self.texts = []
		#self.name2type = dict()

    def load_stopwords(self, in_file):
        with open(in_file) as IN:
            for line in IN:
                self.stopwords.add(line.strip().lower())

    def load_mapping(self, input_file):
        with open(input_file, encoding='utf8') as IN:
            for line in IN:
                tmp = line.strip().split('\t')
                self.name2id[tmp[0]] = int(tmp[1])
                self.id2name[int(tmp[1])] = tmp[0]
        #print(self.name2id['DOC_1000'])

    def dump_mapping(self, output_file):
        with open(output_file, 'w') as OUT:
            for k,v in self.name2id.items():
                OUT.write("{}\t{}\n".format(k,v))

    def translate_emb(self, input_emb, output_emb):
        with open(input_emb) as IN, open(output_emb, 'w', encoding='utf8') as OUT:
            OUT.write(IN.readline())
            for line in IN:
                tmp = line.strip().split(' ')
                if 'P:' in tmp[0]:
                    tmp[0] = str(self.id2name[int(tmp[0].lstrip('P:'))])
                elif 'D:' in tmp[0]:
                    tmp[0] = str(self.id2name[int(tmp[0].lstrip('D:'))])
                OUT.write(' '.join(tmp) + '\n')

    # Normalize text
    def normalize_text(self):
        # Lower case
        texts = [x.lower() for x in self.texts]

        # Remove numbers
        texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

        # Remove stopwords and punctuation
        texts = [' '.join([word.strip(string.punctuation) for word in x.split() if word not in (self.stopwords)]) for x in texts]

        # Trim extra whitespace
        self.texts = [' '.join(x.split()) for x in texts]
        
        #return(texts)

    #add mapping here
    def build_dictionary(self, vocabulary_size = 13000, freq = 10):
        syn_set = {
        'american_football': 'football',
        'association_football': 'soccer',
        'ice_hockey': 'hockey'
        }
        # Turn sentences (list of strings) into lists of words
        split_sentences = [s.split() for s in self.texts]
        split_tuples = [s[:2] for s in self.tuples]

        labels = set(map(lambda x:x[1], self.tuples))
        #print(labels)
        #print(split_sentences, split_tuples)
        split_sentences += split_tuples
        words = [x for sublist in split_sentences for x in sublist]
        
        # Initialize list of [word, word_count] for each word, starting with unknown
        count = [['RARE', -1]]
        
        # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
        #count.extend(collections.Counter(words).most_common(vocabulary_size-1))
        count.extend(collections.Counter(words).most_common(vocabulary_size-1))
        # Now create the dictionary
        # For each word, that we want in the dictionary, add it, then make it
        # the value of the prior dictionary length
        for word, word_count in count:
            #if word_count < freq and word not in labels:
            #    continue
            if word in syn_set:
                #print(word, word_count)
                self.name2id[word] = self.name2id[syn_set[word]]
            else:
                self.name2id[word] = len(self.name2id)
        self.id2name = dict(zip(self.name2id.values(), self.name2id.keys()))
        self.num_words = len(self.name2id)
        #return(self.name2id)

    # Turn text data into lists of integers from dictionary
    def text_to_numbers(self):
        # Initialize the returned data
        self.data = []
        self.tuple_data = []
        for sentence in self.texts:
            sentence_data = []
            # For each word, either use selected index or rare word index
            for word in sentence.split():
                if word in self.name2id:
                    word_ix = self.name2id[word]
                    sentence_data.append(word_ix)
                else:
                    word_ix = 0
                
            self.data.append(sentence_data)
        for tup in self.tuples:
            tup_data = []
            for word in tup[:2]:
                if word in self.name2id:
                    word_ix = self.name2id[word]
                else:
                    print(word)
                    assert True == False
                    word_ix = 0
                tup_data.append(word_ix)
            tup_data.append(tup[-1])
            self.tuple_data.append(tup_data)

    def load_corpus(self, corpusIn, jsonIn, attn = False):
        cnt = 0
        ner_set = defaultdict(int)
        ner_types = set()
        with open(corpusIn) as IN, open(jsonIn) as JSON:
            for cline, jline in tqdm(list(zip(IN.readlines(), JSON.readlines()))):
                if attn:
                    ner = json.loads(jline)['ner']
                    for n in ner:
                        ner_types.add(n[-1])
                        if n[-1] in ['ORDINAL', 'CARDINAL']:
                            continue
                        ner_set[n[0]] += 1
                #else:
                #    d = self.Linker.expand(ner, 1)
                #    self.tuples += d
                self.texts.append(cline)
        if attn:
            filtered = [(k, ner_set[k]) for k in ner_set if ner_set[k] > 10]

            d, t2wid, wid2surface = self.Linker.expand([t[0] for t in filtered], 2)

            stats = defaultdict(int)
            for dd in d:
                stats[(dd[1],dd[2])] += 1
            stats = stats.items()
            hierarchy = Hierarchy(d)
        #print(sorted(stats, key = lambda x:x[1], reverse = True))
        self.num_docs = len(self.texts)

        if attn:
            return hierarchy, t2wid, wid2surface

    def load_concepts(self, corpusIn, concepts):
        with open(corpusIn) as IN:
            for line in tqdm(list(IN.readlines())):
                tokens = line.strip().split(' ')
                for concept in concepts:
                    self.tuples.extend(concept.link_corpus(tokens))

        #for concept in concepts:
        #    print(concept.output_concepts())
        #print(self.tuples)

    # backup functions
    def _load_corpus(self, corpusIn, jsonIn, attn = False):
        cnt = 0
        ner_set = defaultdict(int)
        ner_types = set()
        with open(corpusIn) as IN, open(jsonIn) as JSON:
            for cline, jline in tqdm(list(zip(IN.readlines(), JSON.readlines()))):
            #for cline, jline in zip(IN.readlines(), JSON.readlines()):
                #old freebase Linker
                #ner = list(set(map(lambda x:x[0].strip().replace(' ','_').lower(), json.loads(jline)['ner'])))
                #wikidata Linker
                ner = json.loads(jline)['ner']
                if attn:
                    for n in ner:
                        ner_types.add(n[-1])
                        if n[-1] in ['ORDINAL', 'CARDINAL']:
                            continue
                        ner_set[n[0]] += 1
                else:
                    d = self.Linker.expand(ner, 1)
                    self.tuples += d
                self.texts.append(cline)
        filtered = [(k, ner_set[k]) for k in ner_set if ner_set[k] > 30]

        d, t2wid, wid2surface = self.Linker.expand([t[0] for t in filtered], 2)

        stats = defaultdict(int)
        for dd in d:
            stats[(dd[1],dd[2])] += 1
        stats = stats.items()
        # print(sorted(stats, key = lambda x:x[1], reverse = True))
        self.num_docs = len(self.texts)

        hierarchy = Hierarchy(d)

        return hierarchy

            
    def buildTrain(self, window_size = 1, method='doc2vec'):
        #assert len(self.data) == len(self.kws)
        inputs, outputs = [], []
        for idx, sent in enumerate(self.data):
            if method == 'doc2vec' or method == 'knowledge2vec':
                batch_and_labels = [(sent[i:i+window_size], sent[i+window_size]) for i in range(0, len(sent)-window_size)]
                #print(batch_and_labels)
                try:
                    batch, labels = [list(x) for x in zip(*batch_and_labels)]
                except:
                    continue
                batch = [x + [idx] for x in batch]

            elif 'skip_gram' in method:
                tuple_data = [(y, idx) for y in sent]
                if len(tuple_data) == 0:
                    continue
                batch, labels = zip(*tuple_data)
                batch = [[x] + [idx] for x in batch]
            else:
                window_sequences = [sent[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(sent)]
                # Denote which element of each window is the center word of interest
                label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
                batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
                # Make it in to a big list of tuples (target word, surrounding word)
                
                tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
                if len(tuple_data) == 0:
                    continue
                batch, labels = zip(*tuple_data)

            #add doc index
                batch = [[x] + [idx] for x in batch]
            inputs += batch
            outputs += labels
        #print(outputs[-1])
        print("Training data stats: records {}, kb pairs {}".format(len(inputs), len(self.tuple_data)))
        if method == 'knowledge2vec':
            for tup in self.tuple_data:
                inputs.append([tup[0]]*window_size + [tup[2] + len(self.data)])
                outputs.append(tup[1])
        elif method == 'knowledge2skip_gram':
            for tup in self.tuple_data:
                inputs.append([tup[0], tup[2] + len(self.data)])
                outputs.append(tup[1])

        batch_data = np.array(inputs)
        label_data = np.transpose(np.array([outputs]))
        print("Training data stats: records {}, kb pairs {}".format(batch_data.shape[0], len(self.tuple_data)))
        return batch_data, label_data, self.num_docs, self.num_words

    def buildTrain_(self, num_sampled = 5):
        inputs, outputs = [], []

class Hierarchy:
    def __init__(self, links):
        # filter: whether filter hierarchy based on number of children
        self.links = links
        self.entity_node = set()
        for link in self.links:
            self.entity_node.add((link[0], link[-1] - 1))
            self.entity_node.add((link[1], link[-1]))

        self.p2c = defaultdict(lambda: defaultdict(list))
        self.c2p = defaultdict(lambda: defaultdict(list))
        for link in self.links:
            parent_node = (link[1], link[-1])
            child_node = (link[0], link[-1] - 1)
            relation_type = link[-2]
            self.p2c[parent_node][relation_type].append(child_node)
            self.c2p[child_node][relation_type].append(parent_node)

        self.layer_node = defaultdict(list)
        for entity_node in self.entity_node:
            self.layer_node[entity_node[1]].append(entity_node)

        self.get_leaf_nodes()
        self.get_non_leaf_nodes()

    def save_hierarchy(self, path='/shared/data/qiz3/text_summ/src/jt_code/hierarchies_jt.dump'):
        with open(path, 'wb') as OUT:
            self.p2c = dict(self.p2c)
            self.c2p = dict(self.c2p)
            pickle.dump(self, OUT)

    def filter(self, num):
        keys = self.p2c.keys()
        for key in keys:
            relation_keys = deepcopy(list(self.p2c[key].keys()))
            for relation in relation_keys:
                if len(self.p2c[key][relation]) >= num:
                    continue
                for c in self.p2c[key][relation]:
                    self.c2p[c][relation].remove(key)
                del self.p2c[key][relation]

    def generate_all_top_down_paths(self, entity_node):
        all_paths = set()
        def dfs(current_node, current_path):
            children = self.p2c[current_node]
            if len(children) == 0:
                all_paths.add(deepcopy(current_path))
                return
            for relation in children:
                for child in children[relation]:
                    current_path += (relation,)
                    dfs(child, current_path)
                    current_path = current_path[:-1]

        dfs(entity_node, tuple())

        return all_paths

    def generate_all_concepts(self):
        self.concepts = list()
        for entity_node in self.entity_node:
            if len(self.c2p[entity_node]) > 0:
                continue
            all_paths = self.generate_all_top_down_paths(entity_node)
            for path in all_paths:
                self.concepts.append((entity_node, list(path)))

    def retrieve_nodes_from_concepts(self, concept):
        nodes = [concept[0]]
        for relation in concept[1]:
            new_nodes = []
            for node in nodes:
                if relation in self.p2c[node]:
                    new_nodes.extend(self.p2c[node][relation])
            nodes = new_nodes
        return nodes

    def pick_dense_nodes(self, num):
        node_list = []
        for key in self.p2c:
            for relation in self.p2c[key]:
                if len(self.p2c[key][relation]) >= num:
                    node_list.append((key, relation))

        return node_list

    def get_leaf_nodes(self):
        self.leaves = set()
        for node in self.entity_node:
            if node[1] == 0:
                self.leaves.add(node)

    def get_non_leaf_nodes(self):
        self.non_leaves = set()
        for node in self.entity_node:
            if node[1] > 0:
                self.non_leaves.add(node)

    def count_entity(self, ner_count, t2wid, wid2surface):
        self.entity_count = defaultdict(int)
        layer_num = len(self.layer_node)
        for l in range(layer_num):
            nodes = self.layer_node[l]
            if l == 0:
                for node in nodes:
                    self.entity_count[node] = sum([ner_count[t] for t in wid2surface[t2wid[node[0]]]])
            else:
                for node in nodes:
                    for relation in self.p2c[node]:
                        for c in self.p2c[node][relation]:
                            self.entity_count[node] += self.entity_count[c]

        ranked_hiers = [(k, self.entity_count[k]) for k in self.entity_count]
        ranked_hiers = sorted(ranked_hiers, key=lambda t: -t[1])
        return ranked_hiers

    def ner_linkable(self, ner):
        return (ner, 0) in self.entity_node

    def get_ner_path(self, ner):
        if not self.ner_linkable(ner):
            return None
        all_paths = []
        def dfs(current_node, current_path):
            parents = self.c2p[current_node]
            if len(parents) == 0:
                all_paths.append(deepcopy(current_path))
            for relation in parents:
                for parent in parents[relation]:
                    current_path.append((relation, parent))
                    dfs(parent, current_path)
                    current_path.pop()
        dfs((ner, 0), [(ner, 0)])
        reversed_paths = [(path[-1][1], [t[0] for t in path[:0:-1]]) for path in all_paths]

        return all_paths, reversed_paths

class Concept(object):
    """
    docstring for Concept
    input: [(u'root_node', height), relation_list], e.g. [(u'type of sports',2), [2,4]]
    """
    def __init__(self, hierarchy):
        super(Concept, self).__init__()
        self.hierarchy = hierarchy
        self.links = defaultdict(list)
        #print(self.hierarchy.p2c['type of sport'])
        
    def dfs(self, node, relations, depth):
        for v in self.hierarchy.p2c[node][relations[depth]]:
            #print('{}{}'.format('\t'*(depth+1), v[0].encode('ascii','ignore')))
            self.layers[depth].append(v[0])

            if v[1] > 0:
                self.dfs(v, relations, depth+1)
                #self.links[v[0]].append([node[0], relations[depth]])
                self.links[v[0]].append([node[0], relations[depth]])

    def construct_concepts(self, _concept):
        self.height = len(_concept[1])
        self.layers = [[] for i in range(self.height)]
        self.root = _concept[0]
        relations = _concept[1]
        self.dfs(self.root, relations, 0)

    def output_concepts(self, path):
        output_tuple = []
        with open(path, 'w') as OUTPUT:
            for k,vl in self.links.items():
                for v in vl:
                    if v[0] == self.root[0]:
                        v[0] = k
                    OUTPUT.write("{}|{}\n".format(v[0], k))
                    output_tuple.append([k.lower().replace(' ', '_'), v[0].lower().replace(' ', '_'), v[1]])
        return output_tuple

    def concept_link(self, phrases):
        linked_nodes = set()
        for p in phrases:
            if p in self.links:
                linked_nodes = linked_nodes.union(map(lambda x:x[0], self.links[p]))
        return linked_nodes

    def normalize(self, phrase):
        return phrase.lower().replace(' ', '_')

    def link_corpus(self, phrases):
        syn_set = {
        'football': 'American football',
        'soccer': 'association football',
        'hockey': 'ice hockey'
        }
        linked_nodes = []
        #print(self.links)
        for p in phrases:
            p = p.replace('_', ' ')
            if p in self.links:
                #print(self.links[p])
                #print(list(map(lambda x:[self.normalize(p), self.normalize(x[0]), x[1]], self.links[p])))
                linked_nodes.extend(map(lambda x:[self.normalize(p), self.normalize(p) if x[0] == self.root[0] else self.normalize(x[0]), x[1]], self.links[p]))
        return linked_nodes

def rank_hierarchies(hierarchy, option='A'):
    ranked_list = []
    for node in hierarchy.non_leaves:
        for relation in hierarchy.p2c[node]:
            if len(hierarchy.p2c[node][relation]) <= 3:
                continue
            if option == 'A':
                freqs = [hierarchy.entity_count[t] for t in hierarchy.p2c[node][relation]]
                score = np.mean(freqs)
                #if max(freqs) / sum(freqs) > 0.5:
                #    score = 0
            elif option in ['B', 'C', 'D']:
                freqs = [hierarchy.entity_count[t] for t in hierarchy.p2c[node][relation]]
                avg0 = np.mean(freqs)
                s = sum(freqs)
                freqs = [t / s for t in freqs]
                avg = np.mean(freqs)
                var = np.mean([(t - avg) ** 2 for t in freqs])
                if option == 'B':
                    score = math.log(s) * 1.0 / (var + 0.1)
                elif option == 'C':
                    score = 1.0 / (var + 0.1)
                else:
                    score = avg0 * 1.0 / (var + 0.1)
            elif option == 'E':
                freqs = [hierarchy.entity_count[t] for t in hierarchy.p2c[node][relation]]
                score = sum(freqs)
                freqs = [t / score for t in freqs]
                if max(freqs) > 0.5:
                    score = 0
            else:
                score = 0
            ranked_list.append((node, relation, score))
    ranked_list = sorted(ranked_list, key=lambda t:-t[2])
    return ranked_list

def extract_phrases(target_docs, json_path='/shared/data/qiz3/text_summ/data/NYT_sports.json'):
    JSON = open(json_path)
    ner_cnt = defaultdict(int)
    for idx, jline in tqdm(enumerate(list(JSON.readlines()))):
        if idx not in target_docs:
            continue
        ner = json.loads(jline)['ner']
        for n in ner:
            if n[-1] in ['ORDINAL', 'CARDINAL']:
                continue
            ner_cnt[n[0]] += 1
            #if n[0] in surface2wid and surface2wid[n[0]] in wid2t:
            #    ner_cnt[wid2t[surface2wid[n[0]]]] += 1

    return ner_cnt

def background_assign(concept, t2wid, wid2surface, target_docs = None, json_path='/shared/data/qiz3/text_summ/data/NYT_sports.json'):
    wid2t = {t2wid[k]: k for k in t2wid}
    surface2wid = {t: k for k in wid2surface for t in wid2surface[k]}
    assignment = defaultdict(list)
    target_assigment = []
    JSON = open(json_path)
    for idx, jline in enumerate(tqdm(list(JSON.readlines()))):
        ner_cnt = defaultdict(int)
        ner = json.loads(jline)['ner']
        for n in ner:
            if n[-1] in ['ORDINAL', 'CARDINAL']:
                continue
            ner_cnt[n[0]] += 1
        assigned_node = doc_assign(concept, wid2t, surface2wid, ner_cnt)
        if assigned_node:
            assignment[assigned_node].append(idx)
        if target_docs and idx in target_docs:
            target_assigment.append(assigned_node)

    return assignment, target_assigment

def doc_assign(concepts, docs, label_embeddings):
    top_labels = map(lambda x:x.root[0], concepts)
    candidate_labels = defaultdict(int)
    for doc in docs:
        local_list = []
        for label in top_labels:
            label_vec = label_embeddings[label]
            local_list.append((label, np.dot(vec, label_vec)))
            #local_list.append((label, scipy.spatial.distance.cosine(vec, label_vec)))
        m = max(local_list, key=lambda t:t[1])
        candidate_labels[m] += 1

    main_label = sorted(candidate_labels.items(), key=lambda x:x[1], reverse=True)
    print("Fall into Concept:{}\n".format())

    sub_labels = concepts[top_labels.index()]
    #
    return assigned_node

# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        
        # Pull out center word of interest for each window and create a tuple for each window
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
            batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method=='cbow':
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
            # Only keep windows with consistent 2*window_size
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method=='doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implmented yet.'.format(method))
            
        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return(batch_data, label_data)