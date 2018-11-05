#Preprocessing
import sys
sys.path.append('../')
from FreeBaseLinker import FreebaseLinker
from utils import textGraph
from data import RAEDataset
import torch.utils.data as tdata
from model import KnowledgeD2V
from torch.optim import Adam
from IPython import embed
import torch as t
import numpy as np
import pickle

config = {'batch_size': 128, 'epoch_number': 5, 'emb_size': 100, 'kb_emb_size': 50, 'num_sample': 5, 'gpu':2,
		'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'nyt13_sample', 'method':'doc2vec'}

relation_list=['people.person.nationality','people.person.profession','location.location.contains']
tmp = FreebaseLinker(relation_list)


save_point = pickle.load(open('{}.p'.format(config['dataset']), 'rb'))
X = save_point['X']
y = save_point['y']
num_docs = save_point['num_docs']
num_words = save_point['num_words']

#t.cuda.set_device(int(config['gpu']))

model = KnowledgeD2V(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'],
	kb_emb_size=config['kb_emb_size'], relational_bias=None)
#model.cuda()
model_path = "{}{}_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['epoch_number'])
#model_path = '/shared/data/qiz3/text_summ/src/model/doc2vec_nyt13_110k_epoch_50.pt'
model_ = t.load(model_path, map_location='cpu')
model.load_state_dict(model_, False)

save_path = "{}{}_{}_epoch_{}.npy".format('/shared/data/qiz3/text_summ/output_data/',  config['method'], config['dataset'], config['epoch_number'])
np.save(save_path, model.doc_embeddings())

with open("{}{}_{}_epoch_{}.emb".format('/shared/data/qiz3/text_summ/output_data/',  config['method'], config['dataset'], config['epoch_number']), 'w') as OUT:
	OUT.write("{}\t{}\n".format(num_docs, config['emb_size']))
	for i, line in enumerate(model.doc_embeddings()):
		OUT.write("{}\t{}\n".format(i, ' '.join(map(str, line.tolist()))))