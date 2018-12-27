import sys
from tqdm import tqdm
import pickle
from phraseExtractor import phraseExtractor
sys.path.append('../models/')
import embedder
import torch
import json
from IPython import embed
from collections import defaultdict
import scipy

from model import KnowledgeD2V

def load_mapping(mapping_path):
    # maps phrase to id
    mapping = {}
    with open(mapping_path, 'r') as IN:
        for line in IN:
            phrase, id = line.strip('\n').split('\t')
            mapping[phrase] = id

    return mapping

def load_labels(label_path):
    labels = {}
    candidates = {'College Football': 'football', 'Pro Football': 'football', 'Pro Basketball': 'basketball',
                  'Basketball': 'basketball', 'Hockey': 'hockey', 'Golf': 'golf', 'College Basketball': 'basketball',
                  'Tennis': 'tennis', 'Soccer': 'soccer'}
    with open(label_path) as IN:
        for idx, line in enumerate(IN):
            obj = json.loads(line)
            for l in obj['type']:
                if l.split('/')[-1] not in candidates:
                    continue
                labels[idx] = candidates[l.split('/')[-1]]
            if idx not in labels:
                labels[idx] = None
    return labels

def cos_assign_docs(doc_embeddings, label_embeddings):
    # Use cosine similarity to assign docs to labels
    # doc_embeddings: 2-d numpy array
    # label_embeddings: dict. {'football': vec, ...}
    doc_assignment = defaultdict(list)
    for idx in range(doc_embeddings.shape[0]):
        vec = doc_embeddings[idx]
        local_list = []
        for label in label_embeddings:
            label_vec = label_embeddings[label]
            local_list.append((label, scipy.spatial.distance.cosine(vec, label_vec)))
        m = max(local_list, key=lambda t:t[1])
        doc_assignment[m[0]].append((idx, m[1]))
    return doc_assignment

def evaluate_assignment(doc_assignment, gt_labels, k=20):
    # Evaluate top-k precision
    # doc_assignment: dict, {'football':[0,1,3...], ...}
    # gt_labels: dict, {0: 'football', ...}
    precision = {}
    for label in doc_assignment:
        ranked_list = [t[0] for t in sorted(doc_assignment[label], key=lambda t:-t[1])[k]]
        p = len([1 for x in ranked_list if gt_labels[x] == label]) / k
        precision[label] = p
    return precision


def load_model():
    relation_list=['P54', 'P31', 'P27', 'P641', 'P413', 'P106', 'P1344', 'P17', 'P69', 'P279', 'P463', 'P641']
    config = {'batch_size': 128, 'epoch_number': 50, 'emb_size': 100, 'kb_emb_size': 100, 'num_sample': 5, 'gpu':0,
            'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'NYT_sports', 'method':'knowledge2vec',
            'preprocess': False, 'relation_list':relation_list}

    save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
    num_docs = save_point['num_docs']
    num_words = save_point['num_words']

    model = KnowledgeD2V(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'],
                kb_emb_size=config['kb_emb_size'], relational_bias=config['relation_list'])
    tmp = torch.load('/shared/data/qiz3/text_summ/src/model/knowledge2vec_NYT_sports_epoch_10.pt')
    model.load_state_dict(tmp, False)
    model.cuda()
    return model

if __name__ == '__main__':
    #mapping = load_mapping('/shared/data/qiz3/text_summ/text_summarization/NYT_sports_mapping.txt')
    labels = load_labels('/shared/data/qiz3/text_summ/NYT_sports.json')
    embed()
    exit()
