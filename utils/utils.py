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

class textGraph(object):
    """docstring for textGraph"""
    def __init__(self, arg):
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

    def build_dictionary(self, vocabulary_size = 1000000):
        # Turn sentences (list of strings) into lists of words
        split_sentences = [s.split() for s in self.texts]
        split_tuples = [s[:2] for s in self.tuples]
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
                else:
                    word_ix = 0
                sentence_data.append(word_ix)
            self.data.append(sentence_data)
        for tup in self.tuples:
            tup_data = []
            for word in tup[:2]:
                if word in self.name2id:
                    word_ix = self.name2id[word]
                else:
                    word_ix = 0
                tup_data.append(word_ix)
            tup_data.append(tup[-1])
            self.tuple_data.append(tup_data)

    def load_corpus(self, corpusIn, jsonIn, attn = False):
        #sports = [27, 30, 32, 41, 65, 201, 239, 297, 422, 427, 441, 669, 694, 713, 742, 801, 834, 1006, 1030, 1036, 1119, 1418, 2896, 3353, 3667, 3813, 4367, 4516, 5042, 5638, 6058, 6101, 6469]
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
        #print(self.tuples, ner_set)

        #filtered = [(k, ner_set[k]) for k in ner_set]
        filtered = [(k, ner_set[k]) for k in ner_set if ner_set[k] > 30]
        #filtered = sorted(filtered, key=lambda t:-t[1])

        d, t2wid, wid2original = self.Linker.expand([t[0] for t in filtered], 2)
        #one_hop_expanded = [t[0] for t in d]

        stats = defaultdict(int)
        for dd in d:
            stats[(dd[1],dd[2])] += 1
        stats = stats.items()
        print(sorted(stats, key = lambda x:x[1], reverse = True))
        self.num_docs = len(self.texts)

        #h = self.construct_hierarchies(d)
        #hierarchies = {k: h[k] for k in h if len(h[k]) >= 3}
        #ranked_list = [(k[0], self.cnt_freq(h, k[0], ner_set, 1)) for k in h]
        #ranked_list = sorted(ranked_list, key=lambda t: -t[1])

        hierarchy = Hierarchy(d)

        return hierarchy

    def construct_hierarchies(self, links):
        hierarchies = defaultdict(list)
        for link in links:
            hierarchies[(link[-2], link[-1])].append(link[0])
        #hierarchies = {k: hierarchies[k] for k in hierarchies if len(hierarchies[k]) >= 5}
        return hierarchies

    def cnt_freq(self, hier, entity, ner_cnt, depth):
        if entity not in map(lambda t:t[0], hier.keys()) or depth >= 3:
            return ner_cnt[entity] if entity in ner_cnt else 0
        accum = 0
        for i in range(1, len(self.Linker.relation_list) + 1):
            if not (entity, i) in hier:
                continue
            for child in hier[(entity, i)]:
                accum += self.cnt_freq(hier, child, ner_cnt, depth+1)
        return accum

    # backup functions
    def _load_corpus(self, corpusIn):
        with open(corpusIn) as IN:
            self.texts = IN.readlines()
        self.num_docs = len(self.texts)

            
    def buildTrain(self, window_size = 5, attn = False):
        #assert len(self.data) == len(self.kws)
        inputs, outputs = [], []
        for idx, sent in enumerate(self.data):
            batch_and_labels = [(sent[i:i+window_size], sent[i+window_size]) for i in range(0, len(sent)-window_size)]
            #print(batch_and_labels)
            try:
                batch, labels = [list(x) for x in zip(*batch_and_labels)]
            except:
                continue
            batch = [x + [idx] for x in batch]
            inputs += batch
            outputs += labels
        #print(outputs[-1])
        print("Training data stats: records {}, kb pairs {}".format(len(inputs), len(self.tuple_data)))
        for tup in self.tuple_data:
            inputs.append([tup[0]]*window_size + [tup[2] + len(self.data)])
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

    def count_entity(self, ner_count, t2wid, wid2original):
        self.entity_count = defaultdict(int)
        layer_num = len(self.layer_node)
        for l in range(layer_num):
            nodes = self.layer_node[l]
            if l == 0:
                for node in nodes:
                    self.entity_count[node] = sum([ner_count[t] for t in wid2original[t2wid[node[0]]]])
            else:
                for node in nodes:
                    for relation in self.p2c[node]:
                        for c in self.p2c[node][relation]:
                            self.entity_count[node] += self.entity_count[c]

        ranked_hiers = [(k, self.entity_count[k]) for k in self.entity_count]
        ranked_hiers = sorted(ranked_hiers, key=lambda t: -t[1])
        return ranked_hiers

    def rank_hierarchies(self):
        ranked_list = []
        for node in self.non_leaves:
            for relation in self.p2c[node]:
                if len(self.p2c[node][relation]) <= 3:
                    continue
                ranked_list.append((node, relation, np.mean([self.entity_count[t] for t in self.p2c[node][relation]])))
        ranked_list = sorted(ranked_list, key=lambda t:-t[2])
        return ranked_list

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
        
    def dfs(self, node, relations, depth):
        for v in self.hierarchy.p2c[node][relations[depth]]:
            #print('{}{}'.format('\t'*(depth+1), v[0].encode('ascii','ignore')))
            self.layers[depth].append(v[0])

            if v[1] > 0:
                self.dfs(v, relations, depth+1)
            else:
                self.links[v[0]].append(node[0])

    def construct_concepts(self, _concept):
        self.height = len(_concept[1])
        self.layers = [[] for i in range(self.height)]
        self.root = _concept[0]
        relations = _concept[1]
        self.dfs(self.root, relations, 0)
    
    def concept_link(self, phrases):
        linked_nodes = set()
        for p in phrases:
            if p in self.links:
                linked_nodes = linked_nodes.union(self.links[p])
        return linked_nodes


def doc_assign(hierarchy, target_docs, t2wid, wid2original, json_path='/shared/data/qiz3/text_summ/data/NYT_sports.json'):
    def extract_subnodes(selected_nodes):
        subnodes = deepcopy(selected_nodes)
        current = deepcopy(selected_nodes)
        temp = set()
        while 1:
            for node in current:
                for relation in hierarchy.p2c[node]:
                    for c in hierarchy.p2c[node][relation]:
                        temp.add(c)
            if len(temp) == 0:
                break
            current = deepcopy(temp)
            temp = set()
            subnodes |= current
        return subnodes

    JSON = open(json_path)
    ner_cnt = defaultdict(int)
    wid2t = {t2wid[k]: k for k in t2wid}
    original2wid = {t: k for k in wid2original for t in wid2original[k]}
    for idx, jline in tqdm(enumerate(JSON.readlines())):
        if idx not in target_docs:
            continue
        ner = json.loads(jline)['ner']
        for n in ner:
            if n[-1] in ['ORDINAL', 'CARDINAL']:
                continue
            if n[0] in original2wid and original2wid[n[0]] in wid2t and \
                            (wid2t[original2wid[n[0]]], 0) in hierarchy.entity_node:
                ner_cnt[n[0]] += 1

    entity_count = defaultdict(int)
    selected_nodes = set([('state of the United States', 1), ('type of sport', 2), ('human', 1)])
    layer_num = len(hierarchy.layer_node)
    for l in range(layer_num):
        nodes = hierarchy.layer_node[l]
        if l == 0:
            for node in nodes:
                entity_count[node] = sum([ner_cnt[t] for t in wid2original[t2wid[node[0]]]])
        else:
            for node in nodes:
                for relation in hierarchy.p2c[node]:
                    for c in hierarchy.p2c[node][relation]:
                        entity_count[node] += entity_count[c]

    ranked_list = []
    for node in extract_subnodes(selected_nodes):
        if node[0] == 0 or entity_count[node] == 0:
            continue
        for relation in hierarchy.p2c[node]:
            child_cnt = [entity_count[t] for t in hierarchy.p2c[node][relation]]
            if len(child_cnt) <= 3:
                continue
            score = max(child_cnt)
            ranked_list.append(((node, relation), score))

    ranked_list = sorted(ranked_list, key=lambda t:-t[1])

    embed()
    exit()



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