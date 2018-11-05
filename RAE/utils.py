import string
import os
import json
import io
import tarfile
import collections
import numpy as np

class textGraph(object):
    """docstring for textGraph"""
    def __init__(self, arg):
        super(textGraph, self).__init__()
        self.name2id = dict()
        self.id2name = dict()
        self.stopwords = set()
        self.Linker = arg
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

    def load_corpus(self, corpusIn, jsonIn):
        self.tuples = []
        self.texts = []
        sports = [27, 30, 32, 41, 65, 201, 239, 297, 422, 427, 441, 669, 694, 713, 742, 801, 834, 1006, 1030, 1036, 1119, 1418, 2896, 3353, 3667, 3813, 4367, 4516, 5042, 5638, 6058, 6101, 6469]
        with open(corpusIn) as IN, open(jsonIn) as JSON, open('study.txt', 'w') as study:
            for idx, (cline, jline) in enumerate(zip(IN, JSON)):
                ner = list(set(map(lambda x:x[0].strip().replace(' ','_').lower(), json.loads(jline)['ner'])))
                a,b,c,d = self.Linker.expand(ner, 1)
                if idx in sports:
                    for tup in d:
                        study.write(' '.join(tup[:2]) + '\t')
                    study.write('\n')
                self.tuples += d
                self.texts.append(cline)
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