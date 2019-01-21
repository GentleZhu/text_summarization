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

    def load_corpus(self, corpusIn, jsonIn, relation_cat, reversed_hier, attn = False):
        cnt = 0
        ner_set = defaultdict(int)
        ner_types = set()
        with open(corpusIn) as IN, open(jsonIn) as JSON:
            for cline, jline in tqdm(list(zip(IN.readlines(), JSON.readlines()))[:100]):
                if attn:
                    ner = json.loads(jline)['phrases']
                    for n in ner:
                        #ner_types.add(n[-1])
                        #if n[-1] in ['ORDINAL', 'CARDINAL']:
                        #    continue
                        ner_set[n] += 1
                #else:
                #    d = self.Linker.expand(ner, 1)
                #    self.tuples += d
                self.texts.append(cline)
        if attn:
            filtered = [(k, ner_set[k]) for k in ner_set if ner_set[k] > 0]

            d, t2wid, wid2surface = self.Linker.expand([t[0] for t in filtered], 1)

            stats = defaultdict(int)
            for dd in d:
                stats[(dd[1],dd[2])] += 1
            stats = stats.items()
            hierarchy = Hierarchy(d, relation_cat, reversed_hier)
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
    def __init__(self, links, relation_cat, reversed_hier):
        # filter: whether filter hierarchy based on number of children
        self.links = links
        self.entity_node = set()
        for link in self.links:
            self.entity_node.add((link[0], link[-1] - 1))
            #self.entity_node.add((link[1], link[-1]))

        self.p2c = defaultdict(lambda: defaultdict(list))
        self.c2p = defaultdict(lambda: defaultdict(list))
        for link in self.links:
            parent_node = (relation_cat[link[-2]], link[-1])
            child_node = (link[0], link[-1] - 1)
            relation_type = 0
            if child_node not in self.p2c[parent_node][relation_type]:
                self.p2c[parent_node][relation_type].append(child_node)
            if parent_node not in self.c2p[child_node][relation_type]:
                self.c2p[child_node][relation_type].append(parent_node)

        for c in reversed_hier:
            if (c, 1) in self.p2c:
                if (c, 1) not in self.p2c[(reversed_hier[c], 2)][1]:
                    self.p2c[(reversed_hier[c], 2)][1].append((c, 1))
                if (reversed_hier[c], 2) not in self.c2p[(c, 1)][1]:
                    self.c2p[(c, 1)][1].append((reversed_hier[c], 2))

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
        self.layers = [[]]
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

def load_doc_emb(emb_path):
    with open(emb_path) as EMB:
        headline = EMB.readline().strip().split(' ')
        num_nodes, num_dim = map(int, headline)
        output = np.zeros((num_nodes, num_dim))
        for line in EMB:
            tmp = line.strip().split(' ')
            output[int(tmp[0]), :] = np.asarray(list(map(float, tmp[1:])))
        return output

def load_label_emb(emb_path):
    with open(emb_path) as EMB:
        headline = EMB.readline().strip().split(' ')
        num_nodes, num_dim = headline
        output = dict()
        for line in EMB:
            tmp = line.strip().split(' ')
            output[tmp[0]] = np.asarray(list(map(float, tmp[1:])))
        return output

def background_doc_assign(doc_embeddings, label_embeddings, labels):
    doc_assignment = defaultdict(list)
    for idx in range(doc_embeddings.shape[0]):
        vec = doc_embeddings[idx]
        local_list = []
        for label in labels:
            label_vec = label_embeddings[label]
            local_list.append((label, np.dot(vec, label_vec)))
            #local_list.append((label, scipy.spatial.distance.cosine(vec, label_vec)))
        m = max(local_list, key=lambda t:t[1])
        #if idx > 10:
        #    break
        #print(local_list)
        doc_assignment[m[0]].append((idx, m[1]))
    return doc_assignment

def retrieve_siblings(main_doc_assignment, doc_assignment, labels, topk = 10):
    return_docs = {}
    for label in doc_assignment:
        ranked_list = []
        for t in sorted(doc_assignment[label], key=lambda t:-t[1]):
            if len(ranked_list) >= topk:
                break
            if t[0] in main_doc_assignment:
                ranked_list.append(t[0])
        return_docs[label] = ranked_list
    return return_docs

def target_doc_assign(concepts, docs, label_embeddings, doc_embeddings):
    top_labels = list(map(lambda x:x.root[0], concepts))
    candidate_labels = defaultdict(int)
    target_label = None
    top_labels += ['science', 'business', 'arts', 'politics']
    for doc in docs:
        local_list = []
        vec = doc_embeddings[doc]
        for label in top_labels:
            label_vec = label_embeddings[label]
            local_list.append((label, np.dot(vec, label_vec)))
            #local_list.append((label, scipy.spatial.distance.cosine(vec, label_vec)))
        m = max(local_list, key=lambda t:t[1])[0]
        candidate_labels[m] += 1
    main_label = sorted(candidate_labels.items(), key=lambda x:x[1], reverse=True)[0][0]
    print("Fall into Concept:{}".format(main_label))

    sub_labels = concepts[top_labels.index(main_label)].layers 
    candidate_labels = defaultdict(int)
    for depth in range(len(sub_labels)):
        for doc in docs:
            vec = doc_embeddings[doc]
            local_list = map(lambda x:(x, np.dot(vec, label_embeddings[x])), sub_labels[depth])
            m = max(local_list, key=lambda t:t[1])[0]
            candidate_labels[m] += 1
        entropy = 0
        for m in candidate_labels:
            entropy -= candidate_labels[m] / len(docs) * math.log(candidate_labels[m] / len(docs))
        if entropy > -1:
            print(candidate_labels)
            target_label = max(candidate_labels.items(), key=operator.itemgetter(1))[0]
            print('Entropy is {}, Category:{}'.format(entropy, target_label))
        # now focus on depth-1 node
        break
    #
    return main_label, target_label, sub_labels[depth]

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

def generate_relations():
    all_relations = []
    relations = {}
    relations['Elections'] = ['P541', 'P726', 'P991', 'P1111', 'P1697', 'P1867', 'P1868', 'P2964', 'P2998', 'P3602',
                              'P4247', 'P5043', 'P5044']
    relations['Political_other'] = ['P5045', 'P6', 'P35', 'P122', 'P210', 'P263', 'P530', 'P531', 'P945', 'P1045',
                                    'P1142', 'P1157', 'P1186', 'P1214', 'P1229', 'P1307', 'P1313', 'P1331', 'P1341',
                                    'P1387', 'P1410', 'P1614', 'P1631', 'P1713', 'P1714', 'P1749', 'P1808', 'P1831',
                                    'P1839', 'P1867', 'P1868', 'P1883', 'P1906', 'P1959', 'P1980', 'P1996', 'P2015',
                                    'P2169', 'P2170', 'P2171', 'P2172', 'P2173', 'P2181', 'P2182', 'P2190', 'P2267',
                                    'P2278', 'P2280', 'P2319', 'P2390', 'P2549', 'P2686', 'P2715', 'P2985', 'P2998',
                                    'P3054', 'P3055', 'P3072', 'P3160', 'P3229', 'P3281', 'P3290', 'P3297', 'P3298',
                                    'P3344', 'P3391', 'P3534', 'P3935', 'P3954', 'P4100', 'P4123', 'P4126', 'P4139',
                                    'P4217', 'P4243', 'P4251', 'P4253', 'P4267', 'P4287', 'P4434', 'P4453', 'P4454',
                                    'P4471', 'P4527', 'P4651', 'P4660', 'P4667', 'P4690', 'P4691', 'P4693', 'P4703',
                                    'P4724', 'P4725', 'P4797', 'P4811', 'P4906', 'P4944', 'P4966', 'P4978', 'P4997',
                                    'P5054', 'P5142', 'P5213', 'P5225', 'P5296', 'P5303', 'P5355', 'P5437', 'P5440',
                                    'P5442', 'P5451', 'P5457', 'P5727', 'P5832', 'P5870', 'P5892', 'P6199', 'P6213']
    relations['Photography'] = ['P344', 'P1259', 'P1847', 'P1947', 'P2009', 'P2010', 'P2033', 'P2485', 'P2634', 'P2750',
                                'P3269', 'P4759', 'P5346', 'P5906', 'P6334']
    relations['Music'] = ['P85', 'P86', 'P87', 'P162', 'P175', 'P264', 'P358', 'P406', 'P412', 'P434', 'P435', 'P436',
                          'P483', 'P658', 'P361', 'P676', 'P736', 'P826', 'P839', 'P865', 'P870', 'P942', 'P966',
                          'P1004', 'P1191', 'P1208', 'P1236', 'P1286', 'P1287', 'P1303', 'P1330', 'P1407', 'P1432',
                          'P1553', 'P1558', 'P1625', 'P1725', 'P1728', 'P1729', 'P1730', 'P1731', 'P1762', 'P1763',
                          'P1827', 'P1829', 'P1897', 'P1898', 'P1902', 'P1952', 'P1953', 'P1954', 'P1955', 'P1989',
                          'P1994', 'P2000', 'P2089', 'P2164', 'P2165', 'P2166', 'P2205', 'P2206', 'P2207', 'P2279',
                          'P2281', 'P2291', 'P2336', 'P2338', 'P2373', 'P2510', 'P2513', 'P2514', 'P2550', 'P2624',
                          'P2721', 'P2722', 'P2723', 'P2724', 'P2819', 'P2850', 'P2908', 'P2909', 'P3017', 'P3030',
                          'P3040', 'P3162', 'P3192', 'P3283', 'P3300', 'P3352', 'P3435', 'P3440', 'P3478', 'P3483',
                          'P3511', 'P3674', 'P3733', 'P3736', 'P3763', 'P3838', 'P3839', 'P3854', 'P3952', 'P3977',
                          'P3996', 'P3997', 'P4027', 'P4034', 'P4035', 'P4040', 'P4041', 'P4071', 'P4072', 'P4077',
                          'P4097', 'P4104', 'P4198', 'P4199', 'P4208', 'P4351', 'P4404', 'P4407', 'P4449', 'P4457',
                          'P4497', 'P4518', 'P4576', 'P4577', 'P4578', 'P4579', 'P4603', 'P4607', 'P4747', 'P4748',
                          'P4756', 'P4757', 'P4894', 'P4931', 'P4932', 'P5059', 'P5121', 'P5144', 'P5145', 'P5153',
                          'P5154', 'P5165', 'P5171', 'P5172', 'P5173', 'P5174', 'P5197', 'P5226', 'P5227', 'P5229',
                          'P5235', 'P5240', 'P5241', 'P5251', 'P5261', 'P5262', 'P5272', 'P5287', 'P5291', 'P5292',
                          'P5293', 'P5295', 'P5302', 'P5356', 'P5358', 'P5359', 'P5366', 'P5404', 'P5410', 'P5411',
                          'P5431', 'P5432', 'P5482', 'P5504', 'P5654', 'P5655', 'P5707', 'P5915', 'P5917', 'P5924',
                          'P5927', 'P6013', 'P6079', 'P6080', 'P6091', 'P6110', 'P6111', 'P6116', 'P6122', 'P6130',
                          'P6217', 'P6218', 'P6250']
    relations['Movies'] = ['P58', 'P162', 'P344', 'P345', 'P364', 'P480', 'P905', 'P915', 'P1040', 'P1235', 'P1237',
                           'P1258', 'P1265', 'P1266', 'P1267', 'P1316', 'P1431', 'P1439', 'P1562', 'P1649', 'P1657',
                           'P1712', 'P1804', 'P1874', 'P1934', 'P1969', 'P1970', 'P1981', 'P1985', 'P2019', 'P2168',
                           'P2208', 'P2334', 'P2335', 'P2336', 'P2337', 'P2346', 'P2363', 'P2387', 'P2400', 'P2435',
                           'P2465', 'P2508', 'P2509', 'P2518', 'P2519', 'P2529', 'P2530', 'P2531', 'P2603', 'P2604',
                           'P2605', 'P2626', 'P2629', 'P2631', 'P2636', 'P2637', 'P2639', 'P2678', 'P2684', 'P2688',
                           'P2704', 'P2725', 'P2747', 'P2755', 'P2756', 'P2758', 'P2826', 'P2883', 'P2897', 'P2970',
                           'P3045', 'P3056', 'P3077', 'P3107', 'P3110', 'P3114', 'P3121', 'P3128', 'P3129', 'P3135',
                           'P3136', 'P3138', 'P3139', 'P3140', 'P3141', 'P3142', 'P3143', 'P3144', 'P3145', 'P3146',
                           'P3156', 'P3194', 'P3203', 'P3204', 'P3212', 'P3302', 'P3305', 'P3340', 'P3341', 'P3346',
                           'P3351', 'P3366', 'P3367', 'P3383', 'P3402', 'P3428', 'P3445', 'P3495', 'P3593', 'P3650',
                           'P3673', 'P3703', 'P3704', 'P3750', 'P3785', 'P3786', 'P3787', 'P3803', 'P3804', 'P3808',
                           'P3816', 'P3818', 'P3821', 'P3834', 'P3844', 'P3845', 'P3851', 'P3857', 'P3868', 'P3869',
                           'P3906', 'P3910', 'P3933', 'P3961', 'P3979', 'P3980', 'P3995', 'P4021', 'P4022', 'P4084',
                           'P4085', 'P4086', 'P4129', 'P4270', 'P4276', 'P4277', 'P4282', 'P4283', 'P4312', 'P4326',
                           'P4332', 'P4437', 'P4438', 'P4505', 'P4513', 'P4529', 'P4606', 'P4632', 'P4657', 'P4665',
                           'P4666', 'P4727', 'P4768', 'P4779', 'P4780', 'P4781', 'P4782', 'P4783', 'P4784', 'P4785',
                           'P4786', 'P4834', 'P4910', 'P4947', 'P4981', 'P4983', 'P5032', 'P5033', 'P5083', 'P5091',
                           'P5098', 'P5128', 'P5146', 'P5151', 'P5159', 'P5253', 'P5254', 'P5311', 'P5312', 'P5318',
                           'P5319', 'P5327', 'P5338', 'P5340', 'P5377', 'P5384', 'P5510', 'P5576', 'P5693', 'P5770',
                           'P5786', 'P5791', 'P5820', 'P5865', 'P5885', 'P5925', 'P5932', 'P5941', 'P5970', 'P5987',
                           'P5989', 'P5990', 'P5996', 'P6083', 'P6119', 'P6127', 'P6133', 'P6134', 'P6145', 'P6150',
                           'P6151', 'P6196', 'P6250', 'P6255', 'P6256']
    relations['Fashion'] = ['P2266', 'P2412', 'P2413', 'P2471', 'P2485', 'P2486', 'P2515', 'P2782', 'P3330', 'P3379',
                            'P3404', 'P3482', 'P3812', 'P3814', 'P3828', 'P3832', 'P5995']
    relations['Literature'] = ['P110', 'P123', 'P212', 'P243', 'P356', 'P393', 'P453', 'P648', 'P668', 'P674', 'P1441',
                               'P675', 'P723', 'P840', 'P872', 'P957', 'P1025', 'P1044', 'P1054', 'P1074', 'P1080',
                               'P1084', 'P1085', 'P1104', 'P1112', 'P1143', 'P1144', 'P1165', 'P1182', 'P1253', 'P1274',
                               'P1289', 'P1291', 'P1292', 'P1434', 'P1445', 'P1441', 'P674', 'P1445', 'P1434', 'P1474',
                               'P1564', 'P1565', 'P1574', 'P1575', 'P1608', 'P1685', 'P1739', 'P1802', 'P1815', 'P1823',
                               'P1844', 'P1881', 'P2034', 'P2188', 'P2191', 'P2408', 'P2438', 'P2533', 'P2546', 'P2551',
                               'P2552', 'P2558', 'P2563', 'P2607', 'P2679', 'P2680', 'P2734', 'P2799', 'P2879', 'P2886',
                               'P2963', 'P2969', 'P2973', 'P2977', 'P3004', 'P3092', 'P3154', 'P3155', 'P3184', 'P3277',
                               'P3280', 'P3307', 'P3389', 'P3436', 'P3631', 'P3768', 'P3939', 'P3973', 'P3976', 'P3991',
                               'P4084', 'P4085', 'P4098', 'P4180', 'P4181', 'P4258', 'P4549', 'P4584', 'P4629', 'P4672',
                               'P4675', 'P4717', 'P4749', 'P4881', 'P4907', 'P4928', 'P5039', 'P5101', 'P5123', 'P5341',
                               'P5343', 'P5344', 'P5364', 'P5365', 'P5377', 'P5392', 'P5393', 'P5394', 'P5408', 'P5409',
                               'P5413', 'P5414', 'P5419', 'P5430', 'P5465', 'P5470', 'P5477', 'P5478', 'P5485', 'P5498',
                               'P5502', 'P5503', 'P5506', 'P5509', 'P5527', 'P5532', 'P5538', 'P5539', 'P5542', 'P5543',
                               'P5544', 'P5545', 'P5547', 'P5557', 'P5561', 'P5570', 'P5571', 'P5609', 'P5613', 'P5618',
                               'P5635', 'P5637', 'P5639', 'P5640', 'P5641', 'P5643', 'P5645', 'P5666', 'P5695', 'P5700',
                               'P5705', 'P5710', 'P5711', 'P5712', 'P5714', 'P5747', 'P5752', 'P5792', 'P5800', 'P5940',
                               'P5985', 'P6130', 'P6158', 'P6173', 'P6175', 'P6221', 'P6226', 'P6249']
    relations['Television'] = ['P1267', 'P3121', 'P3156', 'P3804', 'P3970', 'P4022', 'P4270', 'P4312', 'P4400', 'P4438',
                               'P4676', 'P4784', 'P4834', 'P4983', 'P5032', 'P5091', 'P5327', 'P5885', 'P5925', 'P5932']
    relations['Architecture'] = ['P84', 'P149', 'P1014', 'P2194', 'P2385', 'P2418', 'P2917', 'P3058', 'P3960', 'P4206',
                                 'P4304', 'P4488', 'P4534', 'P4694', 'P4980', 'P5308', 'P5383', 'P5508', 'P5573',
                                 'P5604', 'P5745']
    relations['Artworks'] = ['P347', 'P350', 'P1212', 'P1428', 'P1679', 'P1726', 'P2014', 'P2092', 'P2108', 'P2242',
                             'P2282', 'P2344', 'P2511', 'P2539', 'P2582', 'P3272', 'P3293', 'P3386', 'P3467', 'P3504',
                             'P3634', 'P3711', 'P3855', 'P3929', 'P4144', 'P4257', 'P4373', 'P4380', 'P4525', 'P4564',
                             'P4582', 'P4610', 'P4611', 'P4625', 'P4627', 'P4643', 'P4659', 'P4673', 'P4674', 'P4683',
                             'P4684', 'P4686', 'P4692', 'P4701', 'P4704', 'P4709', 'P4712', 'P4713', 'P4721', 'P4737',
                             'P4738', 'P4739', 'P4740', 'P4761', 'P4764', 'P4814', 'P4905', 'P5210', 'P5223', 'P5265',
                             'P5268', 'P5269', 'P5407', 'P5499', 'P5783', 'P5823', 'P5891', 'P6004', 'P6007', 'P6020',
                             'P6141', 'P6152', 'P6238', 'P6239', 'P6246', 'P6310', 'P6332', 'P6355', 'P6356', 'P6358',
                             'P6372', 'P6374']
    relations['Sculptures'] = ['P1656', 'P2380', 'P2914', 'P3386', 'P3467', 'P3711', 'P4721', 'P6141', 'P6238', 'P6239']
    relations['Theatres'] = ['P1191', 'P1217', 'P1218', 'P1219', 'P1242', 'P1362', 'P2340', 'P2468', 'P3820', 'P4079',
                              'P4357', 'P4456', 'P4534', 'P4535', 'P4608', 'P4634', 'P5058', 'P5068', 'P5615', 'P5616',
                             'P5652', 'P5717', 'P5718', 'P5802', 'P5807', 'P5808', 'P5809', 'P5833', 'P5935', 'P5964',
                             'P6113', 'P6132']
    relations['Software'] = ['P123', 'P178', 'P277', 'P306', 'P348', 'P400', 'P404', 'P408', 'P479', 'P751', 'P756',
                             'P852', 'P853', 'P908', 'P914', 'P916', 'P943', 'P1019', 'P1032', 'P1065', 'P1072',
                             'P1073', 'P1195', 'P1324', 'P1372', 'P1401', 'P1414', 'P1547', 'P1712', 'P1733', 'P1845',
                             'P1933', 'P2078', 'P2209', 'P2379', 'P2451', 'P2537', 'P2572', 'P2599', 'P2606', 'P2699',
                             'P2700', 'P2701', 'P2725', 'P2749', 'P2816', 'P2858', 'P2992', 'P3033', 'P3112', 'P3156',
                             'P3180', 'P3374', 'P3418', 'P3435', 'P3442', 'P3454', 'P3473', 'P3483', 'P3499', 'P3511',
                             'P3575', 'P3597', 'P3861', 'P3913', 'P3966', 'P3985', 'P4043', 'P4092', 'P4107', 'P4162',
                             'P4387', 'P4428', 'P4435', 'P4467', 'P4477', 'P4575', 'P4653', 'P4654', 'P4669', 'P4671',
                             'P4685', 'P4705', 'P4710', 'P4742', 'P4763', 'P4769', 'P4776', 'P4806', 'P4809', 'P4810',
                             'P4816', 'P4837', 'P4839', 'P4846', 'P4847', 'P4857', 'P4858', 'P4859', 'P4916', 'P4917',
                             'P4919', 'P4960', 'P4961', 'P4962', 'P4965', 'P4998', 'P5047', 'P5107', 'P5116', 'P5117',
                             'P5118', 'P5119', 'P5196', 'P5247', 'P5290', 'P5305', 'P5345', 'P5360', 'P5367', 'P5371',
                             'P5379', 'P5382', 'P5385', 'P5453', 'P5494', 'P5565', 'P5566', 'P5568', 'P5585', 'P5590',
                             'P5659', 'P5780', 'P5794', 'P5795', 'P5796', 'P5797', 'P5838', 'P5885', 'P5936', 'P5944',
                             'P5971', 'P5999', 'P6032', 'P6068', 'P6078', 'P6170', 'P6172', 'P6182', 'P6197', 'P6229',
                             'P6267', 'P6269', 'P6337', 'P6352']
    relations['Hardware'] = ['P880', 'P1041', 'P1068', 'P1141', 'P1641', 'P2149', 'P2150', 'P2157', 'P2560', 'P2928',
                             'P2935', 'P3559', 'P4653', 'P4654', 'P4788']
    relations['Coporations'] = ['P169', 'P199', 'P249', 'P355', 'P749', 'P414', 'P1128', 'P1278', 'P1454', 'P1955',
                                'P2782', 'P2954', 'P3225', 'P3377', 'P3875', 'P3979', 'P4156', 'P4443', 'P4444',
                                'P4445', 'P4446', 'P4447', 'P4448']
    relations['Manufactures'] = ['P176', 'P186', 'P1056', 'P176', 'P1071', 'P1389', 'P2079', 'P2197', 'P2821', 'P2822',
                                 'P2822', 'P2821', 'P4988', 'P5009', 'P6075', 'P6076']
    relations['Economics_other'] = ['P176', 'P186', 'P1056', 'P176', 'P1071', 'P1389', 'P2079', 'P2197', 'P2821',
                                    'P2822', 'P2822', 'P2821', 'P4988', 'P5009', 'P6075', 'P6076']
    relations['Chemistry'] = ['P117', 'P231', 'P232', 'P233', 'P234', 'P235', 'P246', 'P267', 'P274', 'P352', 'P515',
                              'P556', 'P591', 'P592', 'P595', 'P628', 'P637', 'P639', 'P652', 'P657', 'P660', 'P661',
                              'P662', 'P665', 'P683', 'P695', 'P700', 'P705', 'P715', 'P873', 'P874', 'P875', 'P876',
                              'P877', 'P993', 'P994', 'P995', 'P1033', 'P1086', 'P1108', 'P1109', 'P1117', 'P1121',
                              'P1148', 'P1578', 'P1579', 'P1668', 'P1673', 'P1738', 'P1931', 'P1978', 'P2017', 'P2027',
                              'P2054', 'P2055', 'P2057', 'P2062', 'P2063', 'P2064', 'P2065', 'P2068', 'P2072', 'P2076',
                              'P2077', 'P2083', 'P2084', 'P2085', 'P2086', 'P2101', 'P2102', 'P2107', 'P2113', 'P2115',
                              'P2118', 'P2119', 'P2128', 'P2129', 'P2153', 'P2177', 'P2203', 'P2204', 'P2231', 'P2240',
                              'P2275', 'P2300', 'P2374', 'P2404', 'P2405', 'P2406', 'P2407', 'P2414', 'P2566', 'P2646',
                              'P2658', 'P2665', 'P2710', 'P2712', 'P2717', 'P2718', 'P2840', 'P2874', 'P2877', 'P2926',
                              'P2993', 'P3013', 'P3070', 'P3071', 'P3073', 'P3076', 'P3078', 'P3098', 'P3117', 'P3345',
                              'P3350', 'P3364', 'P3378', 'P3519', 'P3524', 'P3550', 'P3636', 'P3637', 'P3640', 'P3717',
                              'P3771', 'P3772', 'P3773', 'P3774', 'P3775', 'P3776', 'P3777', 'P3778', 'P3779', 'P3780',
                              'P3781', 'P3781', 'P3780', 'P3890', 'P3978', 'P4147', 'P4149', 'P4168', 'P4268', 'P4269',
                              'P4393', 'P4599', 'P4600', 'P4600', 'P4599', 'P4732', 'P4770', 'P4866', 'P4951', 'P4952',
                              'P5000', 'P5040', 'P5041', 'P5042', 'P5219', 'P5220', 'P5926', 'P5929', 'P6185', 'P6272',
                              'P6274']
    relations['Physics'] = ['P105', 'P128', 'P141', 'P171', 'P181', 'P183', 'P225', 'P351', 'P352', 'P353', 'P354',
                            'P405', 'P427', 'P428', 'P486', 'P492', 'P493', 'P494', 'P524', 'P557', 'P563', 'P566',
                            'P2868', 'P574', 'P586', 'P591', 'P592', 'P593', 'P594', 'P595', 'P604', 'P627', 'P636',
                            'P637', 'P638', 'P639', 'P644', 'P645', 'P652', 'P653', 'P662', 'P663', 'P665', 'P667',
                            'P668', 'P671', 'P672', 'P673', 'P680', 'P681', 'P682', 'P684', 'P685', 'P687', 'P688',
                            'P702', 'P689', 'P694', 'P696', 'P697', 'P698', 'P699', 'P702', 'P688', 'P703', 'P704',
                            'P705', 'P715', 'P769', 'P780', 'P783', 'P784', 'P785', 'P786', 'P787', 'P788', 'P789',
                            'P815', 'P830', 'P835', 'P838', 'P842', 'P846', 'P850', 'P923', 'P924', 'P925', 'P926',
                            'P926', 'P925', 'P927', 'P928', 'P938', 'P944', 'P959', 'P960', 'P961', 'P962', 'P970',
                            'P1050', 'P1055', 'P1057', 'P1060', 'P1070', 'P1076', 'P1137', 'P1193', 'P1199', 'P1323',
                            'P1348', 'P1349', 'P1391', 'P1395', 'P1402', 'P1403', 'P2868', 'P1420', 'P1421', 'P1425',
                            'P1461', 'P1531', 'P1550', 'P1554', 'P1582', 'P1672', 'P1583', 'P1603', 'P1604', 'P1605',
                            'P1606', 'P1606', 'P1605', 'P1660', 'P1677', 'P1672', 'P1582', 'P1677', 'P1660', 'P1690',
                            'P1691', 'P1692', 'P1693', 'P1694', 'P1703', 'P1704', 'P1704', 'P1703', 'P1727', 'P1743',
                            'P1744', 'P1745', 'P1746', 'P1747', 'P1748', 'P1761', 'P1772', 'P1832', 'P1843', 'P1895',
                            'P1909', 'P1910', 'P1911', 'P1912', 'P1913', 'P1914', 'P1915', 'P1916', 'P1917', 'P1918',
                            'P1924', 'P1925', 'P1928', 'P1929', 'P1930', 'P1939', 'P1940', 'P1990', 'P1991', 'P1992',
                            'P1995', 'P2006', 'P2007', 'P2008', 'P2024', 'P2026', 'P2036', 'P2040', 'P2074', 'P2143',
                            'P2158', 'P2175', 'P2176', 'P2176', 'P2175', 'P2239', 'P2249', 'P2250', 'P2275', 'P2286',
                            'P2289', 'P2288', 'P2289', 'P2286', 'P2293', 'P2329', 'P2393', 'P2394', 'P2410', 'P2426',
                            'P2433', 'P2434', 'P2455', 'P2464', 'P2520', 'P2542', 'P2548', 'P2576', 'P2597', 'P2646',
                            'P2710', 'P2712', 'P2717', 'P2718', 'P2743', 'P2752', 'P2789', 'P2794', 'P2809', 'P2827',
                            'P2833', 'P2839', 'P2840', 'P2841', 'P2844', 'P2849', 'P2854', 'P2870', 'P2871', 'P2874',
                            'P2892', 'P2941', 'P2944', 'P2946', 'P2974', 'P2975', 'P3031', 'P3060', 'P3063', 'P3064',
                            'P3088', 'P3094', 'P3098', 'P3099', 'P3100', 'P3101', 'P3102', 'P3105', 'P3130', 'P3151',
                            'P3186', 'P3189', 'P3190', 'P3190', 'P3189', 'P3201', 'P3205', 'P3240', 'P3261', 'P3262',
                            'P3262', 'P3261', 'P3288', 'P3289', 'P3291', 'P3292', 'P3310', 'P3322', 'P3329', 'P3331',
                            'P3337', 'P3345', 'P3354', 'P3355', 'P3356', 'P3357', 'P3358', 'P3359', 'P3380', 'P3387',
                            'P3395', 'P3398', 'P3405', 'P3406', 'P3420', 'P3432', 'P3433', 'P3444', 'P3457', 'P3459',
                            'P3464', 'P3465', 'P3485', 'P3486', 'P3487', 'P3488', 'P3489', 'P3490', 'P3491', 'P3492',
                            'P3493', 'P3512', 'P3523', 'P3524', 'P3550', 'P3578', 'P3591', 'P3594', 'P3606', 'P3636',
                            'P3637', 'P3640', 'P3648', 'P3720', 'P3739', 'P3741', 'P3746', 'P3795', 'P3811', 'P3841',
                            'P3852', 'P3853', 'P3860', 'P3870', 'P3885', 'P3937', 'P3945', 'P3951', 'P3956', 'P3982',
                            'P4000', 'P4024', 'P4044', 'P4058', 'P4081', 'P4122', 'P4125', 'P4194', 'P4196', 'P4214',
                            'P4229', 'P4233', 'P4235', 'P4236', 'P4250', 'P4268', 'P4288', 'P4301', 'P4311', 'P4317',
                            'P4333', 'P4338', 'P4355', 'P4394', 'P4395', 'P4425', 'P4426', 'P4433', 'P4495', 'P4526',
                            'P4537', 'P4545', 'P4567', 'P4569', 'P4628', 'P4630', 'P4664', 'P4670', 'P4715', 'P4728',
                            'P4743', 'P4753', 'P4754', 'P4758', 'P4774', 'P4777', 'P4798', 'P4807', 'P4843', 'P4855',
                            'P4863', 'P4864', 'P4866', 'P4873', 'P4875', 'P4882', 'P4902', 'P4914', 'P4915', 'P4954',
                            'P5003', 'P5036', 'P5037', 'P5055', 'P5082', 'P5131', 'P5132', 'P5132', 'P5131', 'P5133',
                            'P5134', 'P5134', 'P5133', 'P5179', 'P5200', 'P5209', 'P5211', 'P5214', 'P5215', 'P5216',
                            'P5219', 'P5221', 'P5230', 'P5231', 'P5248', 'P5257', 'P5263', 'P5270', 'P5299', 'P5315',
                            'P5326', 'P5329', 'P5354', 'P5370', 'P5375', 'P5376', 'P5397', 'P5415', 'P5418', 'P5446',
                            'P5450', 'P5458', 'P5468', 'P5473', 'P5496', 'P5501', 'P5572', 'P5588', 'P5626', 'P5642',
                            'P5683', 'P5698', 'P5806', 'P5818', 'P5841', 'P5843', 'P5858', 'P5862', 'P5864', 'P5878',
                            'P5945', 'P5953', 'P5984', 'P6003', 'P6018', 'P6019', 'P6021', 'P6025', 'P6028', 'P6030',
                            'P6033', 'P6034', 'P6035', 'P6036', 'P6039', 'P6040', 'P6041', 'P6042', 'P6043', 'P6044',
                            'P6045', 'P6046', 'P6047', 'P6048', 'P6049', 'P6050', 'P6051', 'P6052', 'P6053', 'P6054',
                            'P6055', 'P6056', 'P6057', 'P6061', 'P6070', 'P6092', 'P6093', 'P6094', 'P6096', 'P6098',
                            'P6101', 'P6103', 'P6105', 'P6114', 'P6115', 'P6128', 'P6137', 'P6139', 'P6142', 'P6143',
                            'P6159', 'P6161', 'P6163', 'P6176', 'P6177', 'P6209', 'P6220', 'P6227', 'P6245', 'P6264',
                            'P6268', 'P6285', 'P6289', 'P6341', 'P6347', 'P6349', 'P6376']
    relations['Maths'] = ['P549', 'P589', 'P829', 'P889', 'P894', 'P913', 'P1102', 'P1107', 'P1164', 'P1171', 'P1181',
                          'P1247', 'P1312', 'P1678', 'P1322', 'P1470', 'P1556', 'P1563', 'P1568', 'P1571', 'P1678',
                          'P1312', 'P1752', 'P1851', 'P2021', 'P2061', 'P2159', 'P2222', 'P2384', 'P2396', 'P2534',
                          'P2535', 'P2577', 'P2663', 'P2812', 'P2820', 'P2931', 'P3228', 'P3263', 'P3264', 'P3285',
                          'P3457', 'P3492', 'P3752', 'P3753', 'P3754', 'P3755', 'P3756', 'P3757', 'P4242', 'P4252',
                          'P4501', 'P4955', 'P5135', 'P5136', 'P5176', 'P5236', 'P5350', 'P5351', 'P5352', 'P5610',
                          'P5629', 'P5819', 'P5867', 'P5948', 'P6037']
    relations['Geology'] = ['P484', 'P514', 'P534', 'P537', 'P538', 'P565', 'P567', 'P568', 'P568', 'P567', 'P579',
                            'P589', 'P690', 'P693', 'P711', 'P712', 'P713', 'P714', 'P731', 'P732', 'P733', 'P824',
                            'P842', 'P1088', 'P1137', 'P1632', 'P1886', 'P1903', 'P2053', 'P2155', 'P2156', 'P2367',
                            'P2527', 'P2528', 'P2659', 'P2660', 'P2695', 'P2742', 'P2784', 'P2792', 'P3109', 'P3137',
                            'P3196', 'P3309', 'P3507', 'P3513', 'P3514', 'P3635', 'P3770', 'P3813', 'P3815', 'P3907',
                            'P4207', 'P4266', 'P4552', 'P4592', 'P4708', 'P5092', 'P5095', 'P5182', 'P5258', 'P5386',
                            'P5761', 'P5900', 'P6202', 'P6263', 'P6265']
    relations['Astronomy'] = ['P59', 'P65', 'P196', 'P215', 'P223', 'P247', 'P367', 'P376', 'P397', 'P398', 'P398',
                              'P397', 'P399', 'P490', 'P491', 'P522', 'P716', 'P717', 'P720', 'P744', 'P819', 'P824',
                              'P881', 'P999', 'P1016', 'P1046', 'P1090', 'P1096', 'P1145', 'P1215', 'P1227', 'P1457',
                              'P1458', 'P2045', 'P2146', 'P2147', 'P2211', 'P2213', 'P2214', 'P2215', 'P2216', 'P2221',
                              'P2243', 'P2244', 'P2248', 'P2285', 'P2325', 'P2583', 'P2824', 'P2956', 'P3083', 'P3208',
                              'P3592', 'P4095', 'P4296', 'P4341', 'P4384', 'P4466', 'P4501', 'P5650', 'P5653', 'P5667',
                              'P5736', 'P5738', 'P6257', 'P6258', 'P6259', 'P6260', 'P6261', 'P6340']
    for cat in relations:
        all_relations.extend(relations[cat])
    relation_cat = {}
    for cat in relations:
        for r in relations[cat]:
            relation_cat[r] = cat
    upper_hierarchy = {}
    upper_hierarchy['Science'] = ['Astronomy', 'Geology', 'Maths', 'Physics', 'Chemistry']
    upper_hierarchy['Politics'] = ['Elections', 'Political_other']
    upper_hierarchy['Art'] = ['Photography', 'Music', 'Movies', 'Fashion', 'Literature', 'Television', 'Architecture',
                              'Artworks', 'Sculptures', 'Theatres']
    upper_hierarchy['CS'] = ['Software', 'Hardware']
    upper_hierarchy['Economics'] = ['Coporations', 'Manufactures', 'Economics_other']
    reversed_hier = {}
    for p in upper_hierarchy:
        for c in upper_hierarchy[p]:
            reversed_hier[c] = p
    return relation_cat, reversed_hier, all_relations