import sys
from tqdm import tqdm
from data import SummDataset
import torch.utils.data as tdata
from model import KnowledgeD2V
from torch.optim import Adam, SparseAdam, SGD
import torch as t
import pickle


def Train(config, X, y, num_words, num_docs):
	if config['method'] == 'knowledge2vec':
		dataset = SummDataset(X, y)
		data = tdata.DataLoader(dataset, batch_size = config['batch_size'], shuffle=True)

		t.cuda.set_device(int(config['gpu']))

		model = KnowledgeD2V(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'], 
			kb_emb_size=config['kb_emb_size'], relational_bias=config['relation_list'])

		
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
			if epoch % 5 == 0:
				model_path = "{}{}_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], epoch)
				t.save(model.state_dict(), model_path)
			print("Epoch:{}, Loss:{}, Relational Bias:{}".format(epoch, sum(test_loss), sum(relational_bias)))