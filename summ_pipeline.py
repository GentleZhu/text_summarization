import sys, subprocess, re
from utils.WikidataLinker_mg import WikidataLinker
from tqdm import tqdm
from utils.utils import *
import pickle, torch
from phraseExtractor import phraseExtractor
sys.path.append('./models/')
import embedder
from sklearn.feature_extraction.text import TfidfVectorizer
from summarizer import collect_statistics, build_in_domain_dict, generate_caseOLAP_scores, build_co_occurrence_matrix
import summarizer
from models.model import KnowledgeEmbed
import configparser
import torch as t

# Used for tf-idf vectorizer.
def tf_idf_vectorizer(in_file):
    with open(in_file) as IN:
        features = []
        for line in IN:
            line = line.strip('\n').split('\t')
            doc_id, new_passage = int(line[0]), line[1].replace(';', ' ')
            features.append(new_passage)
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(features)
    return feature_vectors, vectorizer

def load_model(config):

    save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
    num_docs = save_point['num_docs']
    num_words = save_point['num_words']
    num_labels = save_point['num_labels']
    

    #num_labels = 29

    model = KnowledgeEmbed(num_words=num_words, num_docs=num_docs, num_labels=num_labels, embed_size=config['emb_size'])
    #model = KnowledgeEmbed(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'],
    #            kb_emb_size=config['kb_emb_size'], relational_bias=config['relation_list'])
    if 'id' not in config:
        model_path = "{}{}_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['epoch_number'])
    else:
        model_path = "{}{}_{}_id_{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['id'], config['epoch_number'])
    tmp = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(tmp, False)
    model.cuda()
    #if config['finetune'] == 'True':
    #	return model, save_point['seq'], save_point['seq_mask']
    #else:
    return model

def expand(label_emb, model, name2id, ranked_list):
	results = []
	for term in ranked_list:
		if term[0] not in name2id:
			print(term[0] + 'not in dic')
			continue
		results.append([term[0], np.dot(label_emb, model.word_embed.weight[name2id[term[0]]].data.cpu().numpy())])
	return sorted(results, key=lambda x:x[1], reverse = True)


def load_config(file_path):
	conf = configparser.ConfigParser()
	conf.read(file_path)
	config = dict(conf['DEFAULT'])
	config['batch_size'] = int(config['batch_size'])
	config['epoch_number'] = int(config['epoch_number'])
	config['emb_size'] = int(config['emb_size'])
	config['kb_emb_size'] = int(config['kb_emb_size'])
	config['num_sample'] = int(config['num_sample'])
	config['gpu'] = int(config['gpu'])
	config['topk'] = int(config['topk'])
	config['preprocess'] = json.loads(config['preprocess'].lower())
	return config

if __name__ == '__main__':
	#Interface of training various embeddings
	config_file = sys.argv[1]
	config = load_config(config_file)
	print(config)
	if config['stage'] == 'train':
		if config['preprocess']:
			graph_builder = textGraph(None)
			graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
			#graph_builder._load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.txt')
			print("Loading Hierarchies")
			text_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.txt'.format(config['dataset'])
			json_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.json'.format(config['dataset'])
			#config['dataset']
			if False:
				h = pickle.load(open("{}_{}_hierarchies.p".format(config['method'], 'NYT_full'), 'rb'))
				sports = pickle.load(open("hierarchy_sports.p", 'rb'))
				h_ = pickle.load(open('topic_hierarchy.p', 'rb'))

				concepts = []

				concept_configs = [[('politics', 2), 2], [('business', 2), 2], [('disaster', 2), 2]]
				for con_config in concept_configs:
					concept = Concept(h_)
					concept.construct_concepts(con_config, True)
					concepts.append(concept)
				#print(concepts[0].links)
				del concepts[0].links['military']
				#concepts[0].links['military'] = ['military_']
				#concepts[0].links['military_'] = ['politics']

				concept = Concept(h)
				concept.construct_concepts([(u'Science', 2), 2])
				concepts.append(concept)

				concept = Concept(sports)
				concept.construct_concepts([(u'type of sport', 2), 2])
				concepts.append(concept)

			'''
			concept = Concept(sports)
			concept.root = (u'type of sport', 2)
			concept.links = {
				'soccer': ['type_of_sport|soccer'],
				'basketball': ['type_of_sport|basketball'],
				'baseball': ['type_of_sport|baseball'],
				'football': ['type_of_sport|football'],
				'tennis': ['type_of_sport|tennis'],
				'golf': ['type_of_sport|golf'],
				'hockey': ['type_of_sport|hockey']
			}
			concepts.append(concept)
			'''
				# concept.output_concepts('concepts.txt')
			concepts = pickle.load(open("{}_{}_concepts.p".format(config['method'], config['dataset']), 'rb'))
			#embed()
			graph_builder.load_corpus(text_path)
			graph_builder.normalize_text()
			keywords = graph_builder.load_concepts(concepts, skip_option = True, expan =True)
			graph_builder.dump_linked_ids("{}_{}_linked_ids.p".format(config['dataset'], config['id']))
			#embed()
			#print(keywords)
			#print(graph_builder.label2id)
			if False:
				for concept in concepts:
					concept.clean_concept()
				pickle.dump(concepts, open("{}_{}_concepts.p".format(config['method'], config['dataset']), 'wb'))
			sys.exit(-1)
			graph_builder.build_dictionary()
			graph_builder.text_to_numbers()
			X, num_docs, num_words, num_labels = graph_builder.buildTrain(method=config['method'])
			
			save_point = {'X': X, 'num_labels':num_labels, 'num_docs':num_docs, 'num_words':num_words}
			
			graph_builder.dump_mapping("{}_mapping.txt".format(config['dataset']))
			graph_builder.dump_label("{}_label.p".format(config['dataset']))
			#graph_builder.dump_linked_ids("{}_linked_ids.p".format(config['dataset']))
			pickle.dump(save_point, open("{}_{}.p".format(config['method'], config['dataset']), 'wb'))
			
		else:
			save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
			X = save_point['X']
			num_docs = save_point['num_docs']
			num_words = save_point['num_words']
			num_labels = save_point['num_labels']
			print(num_docs, num_words, num_labels)

		print("Training Embedding")
		embedder.Train(config, X, [num_words, num_docs, num_labels])
	elif config['stage'] == 'examine':
		# this block is for iteratively training and finetune part 
		
		graph_builder = textGraph()
		graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
		graph_builder.load_label("{}_label.p".format(config['dataset']))
		#graph_builder.load_label("NYT_full_lead-3_example_label.p")
		
		graph_builder.load_linked_ids("{}_{}_linked_ids.p".format(config['dataset'], config['id']))
		concepts = pickle.load(open("{}_{}_concepts.p".format(config['method'], config['dataset']), 'rb'))
		print(graph_builder.label2id)
		#embed()
		sys.exit(-1)
		if True:
			model = load_model(config)
			doc2emb = model.doc_embeddings()
			#print(model.doc_embeddings().shape)
			#print(model.input_embeddings().shape)
			label2emb = dict()
			
			for k in graph_builder.label2id:
			#for k in siblings:
			    label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy() 
			print(len(graph_builder.skip_doc))
			
		else:
			doc2emb = load_doc_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/d_expan.vec')
			label2emb = load_label_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/l_expan.vec')
			phrase2emb = load_phrase_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/p.vec')
		
		doc_assignment,top_label_assignment = soft_assign_docs(doc2emb, label2emb, graph_builder.skip_doc)
		docs_info = []
		with open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json') as meta_json:
			for idx,line in tqdm(enumerate(meta_json)):
				tmp = json.loads(line)
				docs_info.append({'title': tmp['title'], 'type': tmp['type']})
		#top_label_assignment = pickle.load(open('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/src/sib.dump' ,'rb'))
		document_phrase_cnt, inverted_index = collect_statistics('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/nyt_lead_3.txt')

		'''
		for k in top_label_assignment:
			with open('{}{}_epoch{}_{}_docs.txt'.format('/shared/data/qiz3/text_summ/text_summarization/results/', config['id'], config['epoch_number'], k), 'w') as OUT:
				for doc in top_label_assignment[k][:100]:
					docs_info[doc[0]]['score'] = str(doc[1])
					OUT.write(json.dumps(docs_info[doc[0]]) + '\n')
		'''
		#print(top_label_assignment.keys())
		hierarchy = simple_hierarchy()
		for idx,h in enumerate(hierarchy.keys()):
			print(idx, h)
			if h == 'root' :
				continue
			for twin in hierarchy[h]:
				#twin = twin.strip('_').replace('right', 'rights').replace('law', 'law_enforcement')
				docs = [x[0] for x in top_label_assignment[twin][:100]]
				siblings_docs = []
				for sibs in hierarchy[h]:
					#sibs = sibs.strip('_').replace('right', 'rights').replace('law', 'law_enforcement')
					if sibs != twin:
						siblings_docs.append([x[0] for x in top_label_assignment[sibs][:100]])

				phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
				scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index, phrase2idx, option = 'A')
				
				#expan_terms = expand(label2emb[twin], model, graph_builder.name2id, ranked_list[:10])
				if False:
					seed_cnt = 0
					#print(concepts[idx].clean_links)
					for l in concepts[idx].clean_links:
						if concepts[idx].clean_links[l] == twin:
							seed_cnt += 1
					for seed in expan_terms:
						if seed_cnt >= 6:
							break
						if seed[0] not in concepts[idx].clean_links:
							concepts[idx].clean_links[seed[0]] = twin
							seed_cnt += 1
				#print(concepts[idx].clean_links)
				print('concept:{}, category:{}, key phrases:{}'.format(h, twin, ranked_list[:30]))

			embed()
		#pickle.dump(concepts, open("{}_{}_expan_concepts.p".format(config['method'], config['dataset']), 'wb'))
		embed()

		'''
		docs = [x[0] for x in top_label_assignment[twin][:30]]
		siblings_docs = []

		for key in siblings.keys():
			if key != twin:
				siblings_docs.append([x[0] for x in top_label_assignment[key][:100]])
		
		embed()
		'''
	
	elif config['stage'] == 'test':
	# Find concentrated concepts and specific common sense node
		graph_builder = textGraph(None)
		graph_builder.load_linked_ids("{}_linked_ids.p".format(config['dataset']))
		graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
		graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
		graph_builder.load_label("{}_label.p".format(config['dataset']))
		with open(config['input_file']) as IN:
			input_docs = []
			docs = []
			for line in IN:
				line = line.strip()
				phrases = re.findall("<phrase>(.*?)</phrase>", line)
				#print(phrases)
				#result = {}
				for p in set(phrases):
					line = line.replace("<phrase>"+p+"</phrase>", p.replace(' ', '_'))
				input_docs.append(' '.join([p.replace(' ', '_').lower() for p in phrases]))
				docs.append(graph_builder.normalize(line))
				#break
		print(len(input_docs))
		

		feature_vectors, vectorizer = tf_idf_vectorizer('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/full.txt')
		results = summarizer.compare(config, input_docs, feature_vectors, vectorizer, graph_builder.skip_doc)
		#results = summarizer.compare(config, input_docs, feature_vectors, vectorizer)

		
		# OUT = open("intermediate_data/{}_{}_set{}.txt".format(config['summ_method'], config['dataset'], idx), 'w')
		# FILELIST.write("intermediate_data/{}_{}_set{}.txt\n".format(config['summ_method'], config['dataset'], idx))

		# OUT.write(' '.join(map(str, docs)) + '\n')
		# for r in ranked_list:
		# 	OUT.write("{} {}\n".format(r[0], r[1]))
		#sys.exit(-1)

		if True: #KNN module trial
			graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
			graph_builder.load_label("{}_label.p".format(config['dataset']))

			if True:
				model = load_model(config)
				doc2emb = model.doc_embeddings()
				#print(model.doc_embeddings().shape)
				#print(model.input_embeddings().shape)
				label2emb = dict()
				
				for k in graph_builder.label2id:
				#for k in siblings:
				    label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy() 
				print(len(graph_builder.skip_doc))
				
			else:
				doc2emb = load_doc_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/d.vec')
				label2emb = load_label_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/l.vec')

			doc_assignment,top_label_assignment = soft_assign_docs(doc2emb, label2emb, graph_builder.skip_doc)

			count = defaultdict(int)
			for idx in results:
				count[doc_assignment[idx]] += 1
				#print(idx, doc_assignment[idx])
			

			#category = max(count.items(), key=operator.itemgetter(1))[0]
			category = 'election_'
			print("category distribution:{}, inferred topic is {}".format(count, category))

			hierarchy = simple_hierarchy()
			for h in hierarchy:
				if category in hierarchy[h]:
					all_siblings = hierarchy[h]
					break

			siblings_docs = [list(map(lambda x:x[0], top_label_assignment[l][:config['topk']])) for l in all_siblings if l != category]
			twin_docs = list(map(lambda x:x[0], top_label_assignment[category][:config['topk']]))

		document_phrase_cnt, inverted_index = collect_statistics('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/full.txt')
		summarizer.summary(config, docs, siblings_docs, twin_docs, document_phrase_cnt, inverted_index)
			
			#break

		#pickle.dump(phrase_scores, open('baselines/sentence_summ/phrase_scores.p', 'wb'))
	elif config['stage'] == 'finetune':
		t.cuda.set_device(int(config['gpu']))
		if config['preprocess']:
			model = load_model(config)
			text_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.txt'.format(config['dataset'])
			json_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.json'.format(config['dataset'])
			
			
			graph_builder = textGraph()
			graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
			graph_builder.load_label("{}_label.p".format(config['dataset']))
			#print(len(graph_builder.name2id))
			labels = {'physics': 19, 'chemistry': 20, 'astronomy': 21, 'geology': 23, 'biology': 24}
			
			label2emb = dict()
			for k in labels:
				if k in graph_builder.label2id:
					label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy() 
				else:
					print('Missing:',k)
			doc_assignment,top_label_assignment = soft_assign_docs(model.doc_embeddings(), label2emb)

			topk = 500
			classes = list(labels.keys())
			training_data, docs = [], []
			for k in labels:
				label2emb[k] = model.label_embed.weight[labels[k], :].data.cpu().numpy() 
				for _id, score in sorted(top_label_assignment[k])[:topk]:
					#training_data.append([classes.index(k), len(docs)])
					training_data.append(classes.index(k))
					docs.append(_id)

			print(len(docs), docs[0])
			graph_builder.load_corpus(text_path, json_path, attn=False, indices_selected = docs)
			graph_builder.normalize_text()
			save_point = {}
			seq, seq_mask = graph_builder.pad_sequences()
			print(seq.shape, seq_mask.shape)
			save_point['seq'], save_point['seq_mask'] = seq, seq_mask
			save_point['training_data'] = training_data
			save_point['labels'] = list(labels.values())
			pickle.dump(save_point, open("{}_{}_finetune.p".format(config['method'], config['dataset']), 'wb'))
		else:
			model = load_model(config)
			save_point = pickle.load(open("{}_{}_finetune.p".format(config['method'], config['dataset']), 'rb'))
		config['method'] = 'finetune'
		embedder.Train(config, [save_point['seq'], save_point['seq_mask']], [model, save_point['training_data'], save_point['labels']])
		#labels.values()
		#for 
