import sys
from tqdm import tqdm
from data import SummDataset, KnowledgeEmbedDataset
import torch.utils.data as tdata
from model import KnowledgeD2V, KnowledgeSkipGram, KnowledgeEmbed
from torch.optim import Adam, SparseAdam, SGD, Adagrad
import torch as t
import pickle
from IPython import embed


def Train(config, X, params):
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

	elif 'skip_gram' in config['method']:
		dataset = SummDataset(X, y)
		data = tdata.DataLoader(dataset, batch_size = config['batch_size'], shuffle=True)

		t.cuda.set_device(int(config['gpu']))

		model = KnowledgeSkipGram(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'], 
			kb_emb_size=config['kb_emb_size'], relational_bias=config['relation_list'])

		'''
		sparse_optimizer = SGD(params=[{'params': model.word_embed.parameters()},
			{'params': model.out_embed.parameters()},
			{'params': model.doc_embed.parameters()}
			], lr=0.001)
		if model.relational_bias:
			dense_optimizer = SGD(params=[{'params': model.r_linear.parameters()},
				{'params': model.W_r.parameters()}
				], lr=0.001)
		else:
			dense_optimizer = None
		'''
		dense_optimizer = None
		sparse_optimizer = SGD(params=model.parameters(), lr=1)

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
				model_path = "{}{}_{}_id_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['id'], epoch)
				t.save(model.state_dict(), model_path)
			print("Epoch:{}, Loss:{}, Relational Bias:{}".format(epoch, sum(test_loss), sum(relational_bias)))
	elif config['method'] == 'KnowledgeEmbed':
		#print(X[0].shape, X[1].shape)
		train_loader = tdata.DataLoader(
             KnowledgeEmbedDataset(X),
             batch_size=config['batch_size'], shuffle=True, pin_memory=True)
		num_words, num_docs, num_labels = params
		t.cuda.set_device(int(config['gpu']))
		model = KnowledgeEmbed(num_words=num_words, num_docs=num_docs, num_labels=num_labels, embed_size=config['emb_size'])

		optimizer = SGD(params=model.parameters(), lr=0.1)

		model.cuda()

		for epoch in range(config['epoch_number']):
			epoch_loss = 0.0
			for batch_data in tqdm(train_loader):
				#print(batch_data[0][:,0].data.cpu().numpy().tolist())
				loss = model(batch_data, config['num_sample'])
				epoch_loss += loss
				model.zero_grad()
				loss.backward()
				optimizer.step()

			if epoch % 5 == 0:
				model_path = "{}{}_{}_id_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['id'], epoch)
				t.save(model.state_dict(), model_path)
			print("Epoch:{}, Loss:{}".format(epoch, epoch_loss))
			#break
