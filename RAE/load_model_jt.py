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

config = {'batch_size': 128, 'epoch_number': 100, 'emb_size': 100, 'num_sample': 5, 'gpu':1,
		'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'nyt13_sample', 'method':'knowledge2vec'}

relation_list=['people.person.nationality','people.person.profession','location.location.contains']
tmp = FreebaseLinker(relation_list)

### Comment below lines
if True:
	#mapper='/shared/data/qiz3/text_summ/input_data/FB_en_mapper.map'
	#tmp.load_ent_mapper(mapper)
	#tmp.load_k_hop_expansion('/shared/data/qiz3/text_summ/notebook/freebase-small.txt')
	graph_builder = textGraph(tmp)
	graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
	graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/nyt13_110k_summ.txt', '/shared/data/qiz3/text_summ/data/nyt13_110k_summ.json')
	graph_builder.normalize_text()
	graph_builder.build_dictionary()
	graph_builder.text_to_numbers()
	X, y, num_docs, num_words = graph_builder.buildTrain()
	save_point = {'X': X, 'y':y, 'num_docs':num_docs, 'num_words':num_words}
	pickle.dump(save_point, open('nyt_sample.p', 'wb'))
else:
	save_point = pickle.load(open('nyt_sample.p', 'rb'))
	num_docs = save_point['num_docs']
	num_words = save_point['num_words']

t.cuda.set_device(int(config['gpu']))

model = KnowledgeD2V(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'], relational_bias=relation_list)
model.cuda()
model_ = t.load('/shared/data/qiz3/text_summ/src/model/doc2vec_nyt13_110k_epoch_46.pt')
model.load_state_dict(model_, False)

save_path = '/shared/data/qiz3/text_summ/output_data/nyt13_110k_docvecs.npy'
np.save(save_path, model.doc_embeddings())
