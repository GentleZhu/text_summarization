import sys, subprocess, re
from utils.WikidataLinker_mg import WikidataLinker
from tqdm import tqdm
from utils.utils import *
import pickle, torch
from scipy import spatial
from collections import defaultdict
from phraseExtractor import phraseExtractor
sys.path.append('./models/')
import embedder
from sklearn.feature_extraction.text import TfidfVectorizer
from summarizer import collect_statistics, build_background_dict, generate_caseOLAP_scores, _generate_caseOLAP_scores, build_co_occurrence_matrix
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

    save_point = pickle.load(open("{}_{}_expan-{}.p".format(config['method'], config['dataset'], config['expan']), 'rb'))
    num_docs = save_point['num_docs']
    num_words = save_point['num_words']
    num_labels = save_point['num_labels']

    #num_labels = 29

    model = KnowledgeEmbed(num_words=num_words, num_docs=num_docs, num_labels=num_labels, embed_size=config['emb_size'])
    #model = KnowledgeEmbed(num_words=num_words, num_docs=num_docs, embed_size=config['emb_size'],
    #            kb_emb_size=config['kb_emb_size'], relational_bias=config['relation_list'])
    model_path = "{}{}_{}_id_{}_expan-{}_epoch_{}.pt".format(config['model_dir'],  config['method'], config['dataset'], config['id'], config['expan'], config['epoch_number'])
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
		results.append([term[0], term[1] + np.dot(label_emb, model.word_embed.weight[name2id[term[0]]].data.cpu().numpy())])
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
	config['expan'] = int(config['expan'])
	config['input_file'] = config['input_file'].split(',')
	config['preprocess'] = json.loads(config['preprocess'].lower())
	return config

if __name__ == '__main__':
	#Interface of training various embeddings
	config_file = sys.argv[1]
	config = load_config(config_file)
	print(config)
	if config['stage'] == 'train':
		#relation_cat, reversed_hier, relation_list1 = generate_relations()
		#relation_list1 = ['P641']
		#relation_list2 = ['P31']
		if config['preprocess']:
			#embed()
			construct_unified_hierarchy("{}_{}_hierarchies.p".format(config['method'], config['dataset']))
			#tmp = WikidataLinker(relation_list1, relation_list2)
			graph_builder = textGraph(None)
			graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
			print("Loading Hierarchies")
			text_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.txt'.format(config['dataset'])
			json_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.json'.format(config['dataset'])
			#graph_builder._load_corpus(text_path, json_path, relation_cat, reversed_hier, attn=True)
			#exit()
			#config['dataset']
			#h, t2wid, wid2surface = graph_builder._load_corpus(text_path, json_path, relation_cat, reversed_hier,
			#												   attn=True)
			#h.save_hierarchy('tmp_relation_hier.p')
			#exit()
			document_phrase_cnt, inverted_index = collect_statistics(
				'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset']))
			#embed()
			#exit()
			if config['expan'] == 0:
				h = pickle.load(open("{}_{}_hierarchies.p".format(config['method'], config['dataset']), 'rb'))
				#sports = pickle.load(open("hierarchy_sports.p", 'rb'))
				#h = pickle.load(open('tmp_relation_hier.p', 'rb'))

				concepts = []

				
				concept_configs = [[('politics', 2), 2], [('business', 2), 2], [('disaster', 2), 2],
								   [('sports', 2), 2], [('science', 2), 2]]
				for con_config in concept_configs:
					concept = Concept(h)
					concept.construct_concepts(con_config, True)
					concepts.append(concept)
				
				#print(concepts[0].links)
				#TODO(@jingjing): modify the hierarchy, current disaster is not good.
				#del concepts[0].links['military']


				'''
				concept = Concept(h)
				concept.construct_concepts([(u'Science', 2), 2])
				concepts.append(concept)

				concept = Concept(sports)
				concept.construct_concepts([(u'type of sport', 2), 2])
				concepts.append(concept)
				'''
			else:
				concepts = pickle.load(open("{}_{}_expan-{}_concepts.p".format(config['method'], config['dataset'], config['expan']), 'rb'))
			graph_builder.load_corpus(text_path)

			graph_builder.normalize_text()
			#skip_option = True means we only embed documents that have linked phrases
			keywords = graph_builder.load_concepts(concepts, skip_option = True, expan = config['expan'] > 0)
			#it stores documents are not linked in current expansion step
			graph_builder.dump_linked_ids("{}_expan-{}_linked_ids.p".format(config['dataset'], config['expan']))
			
			graph_builder.build_dictionary()
			graph_builder.text_to_numbers()
			X, num_docs, num_words, num_labels = graph_builder.buildTrain(method=config['method'])
			
			save_point = {'X': X, 'num_labels':num_labels, 'num_docs':num_docs, 'num_words':num_words}
			
			graph_builder.dump_mapping("{}_expan-{}_mapping.txt".format(config['dataset'], config['expan']))
			graph_builder.dump_label("{}_expan-{}_label.p".format(config['dataset'], config['expan']))
			pickle.dump(save_point, open("{}_{}_expan-{}.p".format(config['method'], config['dataset'], config['expan']), 'wb'))
			if config['expan'] == 0:
				for concept in concepts:
					concept.clean_concept()

				pickle.dump(concepts, open("{}_{}_expan-{}_concepts.p".format(config['method'], config['dataset'], config['expan']), 'wb'))
				print('First concept dumped!')
		else:
			save_point = pickle.load(open("{}_{}_expan-{}.p".format(config['method'], config['dataset'], config['expan']), 'rb'))
			X = save_point['X']
			num_docs = save_point['num_docs']
			num_words = save_point['num_words']
			num_labels = save_point['num_labels']
			print(num_docs, num_words, num_labels)

		print("Training Embedding")
		embedder.Train(config, X, [num_words, num_docs, num_labels])
	elif config['stage'] == 'examine':
		# this block is for label expansion

		#t.cuda.set_device(int(config['gpu']))
		graph_builder = textGraph()
		#load previous step dictionary mapping
		graph_builder.load_mapping("{}_expan-{}_mapping.txt".format(config['dataset'], config['expan']))
		graph_builder.load_label("{}_expan-{}_label.p".format(config['dataset'], config['expan']))
		
		graph_builder.load_linked_ids("{}_expan-{}_linked_ids.p".format(config['dataset'], config['expan']))
		concepts = pickle.load(open("{}_{}_expan-{}_concepts.p".format(config['method'], config['dataset'], config['expan']), 'rb'))
		#for concept in concepts:
		#	print(concept.output_tuple)
		print('number of skip docs:{}'.format(len(graph_builder.skip_doc)))
		if True:
			model = load_model(config)
			doc2emb = model.doc_embeddings()
			#print(model.doc_embeddings().shape)
			#print(model.input_embeddings().shape)
			label2emb = dict()
			
			for k in graph_builder.label2id:
			#for k in siblings:
				if '_' in k:
					label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy() 
			print(len(graph_builder.skip_doc))

		else:
			doc2emb = load_doc_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/d_expan.vec')
			label2emb = load_label_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/l_expan.vec')
			phrase2emb = load_phrase_emb('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_jt/p.vec')

		if False:
			document_phrase_cnt, inverted_index = collect_statistics(
				'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset']))
			model = load_model(config)
			word_embs = model.input_embeddings()
			phrase2emb = {graph_builder.id2name[x]: word_embs[x]
						  for x in tqdm(range(word_embs.shape[0]))}
			dist_map = pickle.load(open('{}_expan-{}_dist_map.p'.format(config['dataset'], config['expan']), 'rb'))
			doc2emb = np.zeros([167842, 100])
			for d in tqdm(range(doc2emb.shape[0])):
				doc2emb[d] = doc_reweight(phrase2emb, [document_phrase_cnt[d]], dist_map, 'B')

		#embed()
		#exit()

		doc_assignment,top_label_assignment = soft_assign_docs(doc2emb, label2emb, graph_builder.skip_doc)
		# for checking top-k relevant documents
		if False:
			docs_info = []
			with open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json') as meta_json:
				for idx,line in tqdm(enumerate(meta_json)):
					tmp = json.loads(line)
					docs_info.append({'title': tmp['title'], 'type': tmp['type']})
			#top_label_assignment = pickle.load(open('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/src/sib.dump' ,'rb'))
		document_phrase_cnt, inverted_index = collect_statistics(
			'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset']))

		kvs = {}
		for l in top_label_assignment:
			kvs[l] = len(top_label_assignment[l])
			#print(l, len(top_label_assignment[l]))
		print(kvs)

		'''
		for k in top_label_assignment:
			with open('{}{}_epoch{}_{}_docs.txt'.format('/shared/data/qiz3/text_summ/text_summarization/results/', config['id'], config['epoch_number'], k), 'w') as OUT:
				for doc in top_label_assignment[k][:100]:
					docs_info[doc[0]]['score'] = str(doc[1])
					OUT.write(json.dumps(docs_info[doc[0]]) + '\n')
		'''

		for concept in concepts:
			print(concept.clean_links)
		hierarchy = simple_hierarchy()
		for idx,h in enumerate(hierarchy.keys()):
			print(idx, h, concepts[idx].clean_links)
			if h == 'root' :
				continue
			for twin in hierarchy[h]:
				#twin = twin.strip('_').replace('right', 'rights').replace('law', 'law_enforcement')
				docs = [x[0] for x in top_label_assignment[twin][:30]]
				siblings_docs = []
				for sibs in hierarchy[h]:
					#sibs = sibs.strip('_').replace('right', 'rights').replace('law', 'law_enforcement')
					if sibs != twin:
						siblings_docs.append([x[0] for x in top_label_assignment[sibs][:30]])

				phrase2idx, idx2phrase = build_background_dict(docs, document_phrase_cnt)
				#TODO(@jingjing): _generate_caseOLAP_scores can not work currently, because it conflict with generate_caseOLAP_scores in test mode, see PhraseExtractor's todo.
				scores, ranked_list = _generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index, phrase2idx, option = 'A')
				print('concept:{}, category:{}, key phrases:{}'.format(h, twin, ranked_list[:20]))
				if config['expan'] >= 0:
					expan_terms = expand(label2emb[twin], model, graph_builder.name2id, ranked_list[:20])
					seed_cnt = 0
					#print(concepts[idx].clean_links)
					'''
					for l in concepts[idx].clean_links:
						if concepts[idx].clean_links[l] == twin:
							seed_cnt += 1
					'''
					expanded_seeds = []
					for seed in expan_terms:
						if seed_cnt >= 3:
							break
						if seed[0] not in concepts[idx].clean_links:
							concepts[idx].clean_links[seed[0]] = twin
							seed_cnt += 1
							expanded_seeds.append(seed[0])
					print('concept:{}, category:{}, expanded seeds:{}'.format(h, twin, expanded_seeds))
					#print(concepts[idx].clean_links)
			embed()
				

		if True:
			pickle.dump(concepts, open("{}_{}_expan-{}_concepts.p".format(config['method'], config['dataset'], config['expan'] + 1), 'wb'))
		#pickle.dump(concepts, open("{}_{}_expan-concepts.p".format(config['method'], config['dataset']), 'wb'))
		#embed()

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
		graph_builder.load_linked_ids("{}_expan-{}_linked_ids.p".format(config['dataset'], config['expan']))
		graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
		graph_builder.load_mapping("{}_expan-{}_mapping.txt".format(config['dataset'], config['expan']))
		graph_builder.load_label("{}_expan-{}_label.p".format(config['dataset'], config['expan']))
		for in_file in config['input_file']:
			with open(in_file) as IN:
				input_docs = []
				docs = []
				for line in IN:
					#print(line)
					#sys.exit(-1)
					line = line.strip()
					phrases = re.findall("<phrase>(.*?)</phrase>", line)
					#print(phrases)
					#result = {}
					for p in set(phrases):
						line = line.replace("<phrase>"+p+"</phrase>", p.replace(' ', '_'))
					input_docs.append(' '.join([p.replace(' ', '_').lower() for p in phrases]))
					docs.append(graph_builder.normalize(line))
					#break
			print(in_file, len(input_docs), len(docs))
			#continue
			#pickle.dump(docs, open('bams_wildfire.p', 'wb'))
			

			#TODO(@jingjing): add weighted average embedding for comparative search(route 2 in the slides)
			if config['comparative_opt'] == 'knn': #KNN comparative search, route 0 and route 1
				document_phrase_cnt, inverted_index = collect_statistics(
					'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset']))
				#graph_builder.load_mapping("{}_expan-{}_mapping.txt".format(config['dataset'], config['expan']))
				#graph_builder.load_label("{}_label.p".format(config['dataset']))

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

				if config['vec_option'] == 'weighted':
					word_embs = model.input_embeddings()
					phrase2emb = {graph_builder.id2name[x]: word_embs[x]
								  for x in tqdm(range(word_embs.shape[0]))}
					phrase_freq = defaultdict(int)
					for doc in docs:
						for phrase in doc:
							phrase_freq[phrase] += 1
					new_emb = doc_reweight(phrase2emb, phrase_freq, None, 'B')
					results = summarizer.compare(config, None, None, None, new_emb, doc2emb, graph_builder.skip_doc)

				elif config['vec_option'] == 'tfidf':
					feature_vectors, vectorizer = tf_idf_vectorizer(
						'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset']))
					# comparative documents
					results = summarizer.compare(config, input_docs, feature_vectors, vectorizer, None, None,
												 graph_builder.skip_doc)
				# results = summarizer.compare(config, input_docs, feature_vectors, vectorizer)
				# print(results)

				count = defaultdict(int)
				for idx in results:
					count[doc_assignment[idx]] += 1
					#print(idx, doc_assignment[idx])
				

				category = max(count.items(), key=operator.itemgetter(1))[0]
				#category = 'election_'
				print("category distribution:{}, inferred topic is {}".format(count, category))

				hierarchy = simple_hierarchy()
				for h in hierarchy:
					if category in hierarchy[h]:
						all_siblings = hierarchy[h]
						break

				siblings_docs = [list(map(lambda x:x[0], top_label_assignment[l][:config['topk']])) for l in all_siblings if l != category]
				twin_docs = list(map(lambda x:x[0], top_label_assignment[category][:config['topk']]))

			elif config['comparative_opt'] == 'weighted_avg':
				document_phrase_cnt, inverted_index = collect_statistics(
					'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset']))
				model = load_model(config)
				word_embs = model.input_embeddings()
				phrase2emb = {}
				for phrase in inverted_index:
					if not phrase in graph_builder.name2id:
						continue
					phrase2emb[phrase] = word_embs[graph_builder.name2id[phrase]]
				phrase_freqs = []
				for doc in docs:
					phrase_freq = defaultdict(int)
					for phrase in doc:
						phrase_freq[phrase] += 1
					phrase_freqs.append(phrase_freq)

				doc2emb = model.doc_embeddings()
				label2emb = dict()

				for k in graph_builder.label2id:
					# for k in siblings:
					label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy()

				document_phrase_cnt, inverted_index = collect_statistics(
					'/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/{}.txt'.format(config['dataset'].replace('_lead-3', '')))
				#graph_builder.load_mapping("{}_expan-{}_mapping.txt".format(config['dataset'], config['expan']))
				#graph_builder.load_label("{}_label.p".format(config['dataset']))

				doc_assignment, top_label_assignment = soft_assign_docs(doc2emb, label2emb, graph_builder.skip_doc)

				
				if config['preprocess']:
					#############################################
					# Used for calculating dist map and doc_emb #
					#############################################
					new_labels = []
					for label in label2emb:
						if '_' in label:
							new_labels.append(label)
					print(new_labels)
					new_doc2emb = {str(i): doc2emb[i] for i in range(doc2emb.shape[0]) if i not in graph_builder.skip_doc}
					dist_map = calculate_distinct_map_1(new_labels, new_doc2emb, phrase2emb, label2emb, inverted_index,
														'soft')
					pickle.dump(dist_map,
								open('dist_map_{}_{}_expan-{}.p'.format(config['method'], config['dataset'],
																		config['expan']),
									 'wb'))
					new_doc_emb = {}
					for d in tqdm(range(doc2emb.shape[0])):
						new_doc_emb[d] = doc_reweight(phrase2emb, [document_phrase_cnt[d]], dist_map, 'B')
					pickle.dump(dist_map, open('{}_expan-{}_dist_map.p'.format(config['dataset'], config['expan']), 'wb'))
					pickle.dump(new_doc_emb,
								open('{}_expan-{}_bg_doc2emb.p'.format(config['dataset'], config['expan']), 'wb'))
				else:
					dist_map = pickle.load(open('{}_expan-{}_dist_map.p'.format(config['dataset'], config['expan']), 'rb'))
					new_doc_emb = pickle.load(open('{}_expan-{}_bg_doc2emb.p'.format(config['dataset'], config['expan']), 'rb'))

				new_emb = doc_reweight(phrase2emb, phrase_freqs, dist_map, 'B')

				if False:
					new_doc_emb = {}
					related_docs = []
					with open('../data/NYT_annotated_corpus/wp_1617.txt', 'r') as IN:
						for idx,line in enumerate(IN):
							phrase_freq = defaultdict(int)
							for phrase in line:
								phrase_freq[phrase] += 1
							related_docs.append(line.strip())
							new_doc_emb[idx] = doc_reweight(phrase2emb, [phrase_freq], dist_map, 'B')
				
				hierarchy = simple_hierarchy()

				if False:
					count = defaultdict(int)
					for doc_agg_emb in new_emb:
						sim_max = -1
						category = None
						for label in label2emb:
							if '_' not in label:
								continue
							sim = 1 - spatial.distance.cosine(doc_agg_emb, label2emb[label])
							if sim > sim_max:
								sim_max = sim
								category = label
						count[category]+= 1
					print(count)
					
					picked_category = max(count.items(), key=operator.itemgetter(1))[0]
					print('New docs assigned to ' + picked_category)
				else:
					hierarchy['root'] = ['disaster', 'science', 'politics', 'business', 'sports']
					category,sub_cat = target_hier_doc_assign_top_down(hierarchy, new_emb, label2emb)
					print('inferred category:{}'.format(category))
				#embed()
				#category = 'election_'
				#sub_cat = []

				if len(sub_cat) == 0:
					for h in hierarchy:
						if category in hierarchy[h]:
							all_siblings = hierarchy[h]
							break
					#
					siblings_docs = [list(map(lambda x: x[0], top_label_assignment[l][:config['topk']])) for l in all_siblings
									 if l != category]
					twin_docs = list(map(lambda x: x[0], top_label_assignment[category][:config['topk']]))

					comp_pool = list(map(lambda x: x[0],top_label_assignment[category]))
				else:
					assert category in ['disaster', 'science', 'politics', 'business', 'sports']
					twin_docs = []
					comp_pool = []
					for cat in sub_cat:
						twin_docs += list(map(lambda x: x[0], top_label_assignment[cat][:config['topk']]))
						comp_pool += list(map(lambda x: x[0],top_label_assignment[cat]))
					siblings_docs = []
					for _category in hierarchy['root']:
						tmp_docs = []
						if category == _category:
							continue
						for cat in hierarchy[_category]:
							tmp_docs += list(map(lambda x: x[0], top_label_assignment[cat][:config['topk']]))
						siblings_docs.append(tmp_docs) 


				comparative_docs = summarizer.compare(config, None, None, None, new_emb, new_doc_emb, graph_builder.skip_doc, comp_pool)
				#comparative_docs = summarizer.compare(config, None, None, None, new_emb, new_doc_emb)
				
				if 'NYT' in config['dataset']:
					docs_info = []
					with open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json') as meta_json:
						for idx,line in tqdm(enumerate(meta_json)):
							tmp = json.loads(line)
							docs_info.append({'title': tmp['title'], 'type': tmp['type']})

				#embed()
					if False:
						with open('tmp_{}.txt'.format(config['input_file'].lstrip('results/').rstrip('.txt')), 'w') as OUT:
							for d in comparative_docs:
								OUT.write(docs_info[d]['title'] + '\n')
				else:
					docs_info = []
					with open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/washington_full.txt') as meta_json:
						for idx,line in tqdm(enumerate(meta_json)):
							docs_info.append({'title': line})

				#embed()
					with open('tmp_{}.txt'.format(config['input_file'].lstrip('results/').rstrip('.txt')), 'w') as OUT:
						for d in comparative_docs:
							OUT.write(docs_info[d]['title'] + '\n')

			
			summarizer.summary(config, docs, siblings_docs, twin_docs, comparative_docs, document_phrase_cnt, inverted_index, graph_builder=graph_builder, file_name=in_file)
			
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
