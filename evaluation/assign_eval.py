import sys
from tqdm import tqdm
import pickle
sys.path.append('./models/')
sys.path.append('./utils/')
from utils import textGraph
import torch
import json
from IPython import embed
from collections import defaultdict
import scipy.spatial
import numpy as np
import math

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
    #candidates = {'College Football': 'football', 'Pro Football': 'football', 'Pro Basketball': 'basketball',
    #              'Basketball': 'basketball', 'Hockey': 'hockey', 'Golf': 'golf', 'College Basketball': 'basketball',
    #              'Tennis': 'tennis', 'Soccer': 'soccer', 'Baseball': 'baseball'}
    #candidates = {'College Football': 'Sports|Football', 'Pro Football': 'Sports|Football', 'Pro Basketball': 'Sports|Basketball',
    #              'Basketball': 'Sports|Basketball', 'Hockey': 'Sports|Hockey', 'Golf': 'Sports|Golf', 'College Basketball': 'Sports|Basketball',
    #              'Tennis': 'Sports|Tennis', 'Soccer': 'Sports|Soccer', 'Baseball': 'Sports|Baseball'} 
    candidates = {'College Football': 'type_of_sport|football', 'Pro Football': 'type_of_sport|football', 'Pro Basketball': 'type_of_sport|basketball',
                  'Basketball': 'type_of_sport|basketball', 'Hockey': 'type_of_sport|hockey', 'Golf': 'type_of_sport|golf', 'College Basketball': 'type_of_sport|basketball',
                  'Tennis': 'type_of_sport|tennis', 'Soccer': 'type_of_sport|soccer', 'Baseball': 'type_of_sport|baseball'}             
    counts = defaultdict(int)
    cnt = 0
    with open(label_path) as IN:
        for idx, line in enumerate(IN):
            obj = json.loads(line)
            for l in obj['type']:
                if l.split('/')[-1] not in candidates:
                    continue
                labels[idx] = candidates[l.split('/')[-1]]
                counts[candidates[l.split('/')[-1]]] += 1
            if idx not in labels:
                #cnt +=1
                labels[idx] = None
    #print("Ground Truth Counts:", counts)
    #print(labels)
    return labels, set(candidates.values()), counts

def cossim(p, q):
    if len(p) != len(q):
        print('KL divergence error: p, q have different length')
    
    p_len = q_len = mix_len = 0

    for i in range(len(p)):
        mix_len += p[i] * q[i]
        p_len += p[i] * p[i]
        q_len += q[i] * q[i]

    return mix_len / (math.sqrt(p_len) * math.sqrt(q_len))

# TODO: write an interface in each model
def cos_assign_docs(doc_embeddings, label_embeddings, gt_labels=None):
    # Use cosine similarity to assign docs to labels
    # doc_embeddings: 2-d numpy array
    # label_embeddings: dict. {'football': vec, ...}
    doc_assignment = defaultdict(list)
    per_doc_assignment = defaultdict(str)
    for idx in range(doc_embeddings.shape[0]):
        if gt_labels and gt_labels[idx] == None:
            continue
        vec = doc_embeddings[idx]
        local_list = []
        for label in label_embeddings:
            label_vec = label_embeddings[label]
            local_list.append((label, np.dot(vec, label_vec)))
            #local_list.append((label, scipy.spatial.distance.cosine(vec, label_vec)))
        m = max(local_list, key=lambda t:t[1])
        #if idx > 10:
        #    break
        #print(local_list)
        doc_assignment[m[0]].append((idx, m[1]))
        per_doc_assignment[idx] = m[0]
    return doc_assignment, per_doc_assignment

def evaluate_assignment(doc_assignment, gt_labels, k=100):
    # Evaluate top-k precision
    # doc_assignment: dict, {'football':[0,1,3...], ...}
    # gt_labels: dict, {0: 'football', ...}
    precision = {}
    for label in doc_assignment:
        ranked_list = [t[0] for t in sorted(doc_assignment[label], key=lambda t:-t[1])[:k]]
        #print(label, [gt_labels[x] for x in ranked_list])
        p = len([1 for x in ranked_list if gt_labels[x] == label]) / k
        precision[label] = p
    return precision

def evaluate_assignment_all(doc_assignment, gt_labels, gt_counts):
    prediction, correct = defaultdict(int), defaultdict(int)
    for idx in doc_assignment:
        prediction[doc_assignment[idx]] += 1
        if doc_assignment[idx] == gt_labels[idx]:
            correct[doc_assignment[idx]] += 1
        #precision[label] = p
    macroF1 = dict()
    for l in prediction:
        if correct[l] == 0:
            continue
        prec, recall = correct[l]/prediction[l], correct[l]/gt_counts[l]
        print("label:{}, precision:{}, recall:{}".format(l, prec, recall))
        macroF1[l] = 2*prec*recall / (prec + recall)
    prec = sum(correct.values()) / sum(prediction.values())
    recall = sum(correct.values()) / sum(gt_counts.values())
    print("MicroF1: {}, MacroF1: {}".format(2*prec*recall / (prec+recall), sum(macroF1.values()) / len(macroF1)))
    return correct


def load_model(config):

    save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
    num_docs = save_point['num_docs']
    num_words = save_point['num_words']

    model = KnowledgeD2V(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'],
                kb_emb_size=config['kb_emb_size'], relational_bias=config['relation_list'])
    if 'id' not in config:
        model_path = "{}{}_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['epoch_number'])
    else:
        model_path = "{}{}_{}_id_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['id'], config['epoch_number'])
    tmp = torch.load(model_path)
    model.load_state_dict(tmp, False)
    model.cuda()
    return model

def load_emb(emb_path):
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

#report micro-precision, micro/macro F1

if __name__ == '__main__':
    relation_list=['P54', 'P31', 'P27', 'P641', 'P413', 'P106', 'P1344', 'P17', 'P69', 'P279', 'P463', 'P641']
    #config = {'batch_size': 128, 'epoch_number': 40, 'emb_size': 100, 'kb_emb_size': 100, 'num_sample': 5, 'gpu':0,
    #    'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'NYT_sports', 'method':'knowledge2skip_gram', 'id':3,
    #    'preprocess': True, 'relation_list':[]}
    config = {'doc_emb_path': 'baselines/doc2cube/tmp/d.vec', 'dataset':'NYT_sports', 'method':'doc2cube', 
    'label_emb_path':'baselines/doc2cube/tmp/l.vec'}
    
    if 'knowledge' not in config['method']:
        config['relation_list'] = []
    
    gt_labels, labels, gt_counts = load_labels('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json')
    print(labels, gt_counts)
    if config['method'] != 'doc2cube':
        graph_builder = textGraph()
        graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
        model = load_model(config)
        assert(model.doc_embeddings().shape[0] == len(gt_labels))
        print(model.input_embeddings().shape)
        label2emb = dict()
        for k in labels:
            if k in graph_builder.name2id:
                label2emb[k] = model.word_embed.weight[graph_builder.name2id[k], :].data.cpu().numpy() 
            else:
                print('Missing:',k)
        
        doc_assignment,per_doc_assignment =  cos_assign_docs(model.doc_embeddings(), label2emb, gt_labels)
    else:
        doc_embeddings = load_emb(config['doc_emb_path'])
        #print(doc_embeddings.shape)
        #print(len(gt_labels))
        assert(doc_embeddings.shape[0] == len(gt_labels))
        label2emb = load_label_emb(config['label_emb_path'])
        doc_assignment,per_doc_assignment =  cos_assign_docs(doc_embeddings, label2emb, gt_labels)
    prec = evaluate_assignment(doc_assignment, gt_labels)
    evaluate_assignment_all(per_doc_assignment, gt_labels, gt_counts)
    print(prec)
