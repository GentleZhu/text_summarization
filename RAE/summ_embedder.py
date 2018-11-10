#Preprocessing
import sys
sys.path.append('../')
from FreeBaseLinker import FreebaseLinker
from WikidataLinker_mg import WikidataLinker
from tqdm import tqdm
from utils import textGraph
from data import RAEDataset
import torch.utils.data as tdata
from model import KnowledgeD2V
from torch.optim import Adam, SparseAdam
import torch as t
import pickle

config = {'batch_size': 128, 'epoch_number': 50, 'emb_size': 100, 'kb_emb_size': 100, 'num_sample': 5, 'gpu':0,
		'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'NYT_sports', 'method':'knowledge2vec'}

#relation_list=['people.person.nationality','people.person.profession','location.location.contains']
relation_list=['P54', 'P31', 'P27', 'P641', 'P413', 'P106', 'P1344', 'P17', 'P69']
#tmp = FreebaseLinker(relation_list)

#P54 team P31 instance of P27 nationality P641 sports P413 position
#P106 occupation P1344 participant P17 country P69 educate P647 draft at

### Comment below lines 
if False:
	#mapper='/shared/data/qiz3/text_summ/input_data/FB_en_mapper.map'
	#tmp.load_ent_mapper(mapper)
	tmp = WikidataLinker(relation_list)
	graph_builder = textGraph(tmp)
	graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
	graph_builder._load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.txt')
	#graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.token', '/shared/data/qiz3/text_summ/data/NYT_sports.json')
	graph_builder.normalize_text()
	graph_builder.build_dictionary()
	graph_builder.text_to_numbers()
	X, y, num_docs, num_words = graph_builder.buildTrain()
	save_point = {'X': X, 'y':y, 'num_docs':num_docs, 'num_words':num_words}
	graph_builder.dump_mapping("{}_mapping.txt".format(config['dataset']))
	pickle.dump(save_point, open("{}_{}.p".format(config['method'], config['dataset']), 'wb'))

else:
	save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
	X = save_point['X']
	y = save_point['y']
	num_docs = save_point['num_docs']
	num_words = save_point['num_words']

print("There are {} documents, and {} words in total".format(num_docs, num_words))
dataset = RAEDataset(X, y)
data = tdata.DataLoader(dataset, batch_size = config['batch_size'], shuffle=True)

t.cuda.set_device(int(config['gpu']))

model = KnowledgeD2V(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'], 
	kb_emb_size=config['kb_emb_size'], relational_bias=relation_list)
sparse_optimizer = SparseAdam(params=[{'params': model.word_embed.parameters()},
	{'params': model.out_embed.parameters()},
	{'params': model.doc_embed.parameters()}
	], lr=0.01)
if model.relational_bias:
	dense_optimizer = Adam(params=[{'params': model.r_linear.parameters()},
		{'params': model.W_r.parameters()}
		], lr=0.01)
else:
	dense_optimizer = None
model.cuda()
for epoch in range(config['epoch_number']):
	test_loss, relational_bias = [], []
	for batch_data in tqdm(data):
		inputs, labels = batch_data
		test_loss_temp, relational_bias_temp = model(inputs, labels, config['num_sample'])
		test_loss.append(test_loss_temp)
		relational_bias.append(relational_bias_temp)
		model.zero_grad()
		test_loss_temp.backward()
		sparse_optimizer.step()
		if dense_optimizer:
			dense_optimizer.step()
	if epoch % 2 == 0:
		model_path = "{}{}_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], epoch)
		t.save(model.state_dict(), model_path)
	print("Epoch:{}, Loss:{}, Relational Bias:{}".format(epoch, sum(test_loss), sum(relational_bias)))