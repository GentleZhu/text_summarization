import sys
from utils.WikidataLinker_mg import WikidataLinker
from tqdm import tqdm
from utils.utils import *
import pickle, torch
from phraseExtractor import phraseExtractor
sys.path.append('./models/')
import embedder
from summarizer import collect_statistics, build_in_domain_dict, generate_caseOLAP_scores, build_co_occurrence_matrix
from summarizer import textrank, ensembleSumm, seedRanker, GCNRanker
from models.model import KnowledgeEmbed

#relation_list=['P54', 'P31', 'P27', 'P641', 'P413', 'P106', 'P1344', 'P17', 'P69', 'P279', 'P463', 'P641']
# relation_list1: hop=1, relation_list2: hop>1
relation_cat, reversed_hier, relation_list1 = generate_relations()
#relation_list1= ['P85', 'P86', 'P87', 'P162', 'P175', 'P264', 'P358', 'P406', 'P412', 'P434', 'P658', 'P676', 'P870', 'P942', 'P1191', 'P1303']
relation_list1 = ['P641']
relation_list2 = ['P31']#, 'P279', 'P361']
config = {'batch_size': 128, 'epoch_number': 0, 'emb_size': 100, 'kb_emb_size': 100, 'num_sample': 5, 'gpu':2,
		'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'NYT_full', 'method':'KnowledgeEmbed', 'id':'jan29',
		'preprocess': True, 'relation_list1':relation_list1, 'relation_list2': relation_list2,
		  'doc_emb_path': 'intermediate_data/pretrain_doc.emb', 'label_emb_path': 'intermediate_data/pretrain_label.emb',
		  'stage': 'test', 'summ_method': 'caseOLAP', 'topk':100}

#config['method'] = 'knowledge2skip_gram'
#config['dataset'] = 'NYT_sports'

#id:0 all information
#id:1 simple label

#P54 team P31 instance of P27 nationality P641 sports P413 position
#P106 occupation P1344 participant P17 country P69 educate P279 subclass of P463 member of P641 sport
def load_gt_labels(label_path):
    target_domain = []
    candidates = {'College Football': 'type_of_sport|football', 'Pro Football': 'type_of_sport|football', 'Pro Basketball': 'type_of_sport|basketball',
                  'Basketball': 'type_of_sport|basketball', 'Hockey': 'type_of_sport|hockey', 'Golf': 'type_of_sport|golf', 'College Basketball': 'type_of_sport|basketball',
                  'Tennis': 'type_of_sport|tennis', 'Soccer': 'type_of_sport|soccer', 'Baseball': 'type_of_sport|baseball'}             
    counts = defaultdict(int)
    cnt = 0
    with open(label_path) as IN:
        for idx, line in enumerate(IN):
            obj = json.loads(line)
            for l in obj['type']:
            	if l.split('/')[-1] in candidates:
                	target_domain.append(idx)
                	continue
                
    #print("Ground Truth Counts:", counts)
    #print(labels)
    return target_domain

def load_model(config):

    save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
    num_docs = save_point['num_docs']
    num_words = save_point['num_words']
    num_labels = save_point['num_labels']

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
    return model

if __name__ == '__main__':
	#Interface of training various embeddings
	if config['stage'] == 'train':
		if config['preprocess']:
			graph_builder = textGraph(None)
			graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
			#graph_builder._load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.txt')
			print("Loading Hierarchies")
			text_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.txt'.format(config['dataset'])
			json_path = '/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/{}.json'.format(config['dataset'])
			h = pickle.load(open("{}_{}_hierarchies.p".format(config['method'], config['dataset']), 'rb'))
			sports = pickle.load(open("hierarchy_sports.p", 'rb'))
			#h = pickle.load(open("{}_{}_hierarchies.p".format(config['method'], config['dataset']), 'rb'))

			concepts = []

			concept_configs = [[(u'Science', 2), 2], [(u'Economics', 2), 2], [(u'Politics', 2), 2]]
			for con_config in concept_configs:
				concept = Concept(h)
				concept.construct_concepts(con_config)
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
			graph_builder.load_corpus(text_path, json_path, attn=False)
			graph_builder.normalize_text()
			graph_builder.load_concepts(concepts)
			print(graph_builder.label2id)
			graph_builder.build_dictionary()
			graph_builder.text_to_numbers()
			X, num_docs, num_words, num_labels = graph_builder.buildTrain(method=config['method'])			
			save_point = {'X': X, 'num_labels':num_labels, 'num_docs':num_docs, 'num_words':num_words}
			graph_builder.dump_mapping("{}_mapping.txt".format(config['dataset']))
			graph_builder.dump_label("{}_label.p".format(config['dataset']))
			pickle.dump(save_point, open("{}_{}.p".format(config['method'], config['dataset']), 'wb'))
		else:
			save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
			X = save_point['X']
			num_docs = save_point['num_docs']
			num_words = save_point['num_words']
			num_labels = save_point['num_labels']
		
		print("Training Embedding")
		embedder.Train(config, X, [num_words, num_docs, num_labels])
	elif config['stage'] == 'examine':
		twin = 'astronomy'
		siblings = {'geology': 0, 'physics': 2, 'chemistry': 3, 'biology': 4, 'maths': 5, 'astronomy': 6}.keys()
		graph_builder = textGraph()
		graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
		graph_builder.load_label("{}_label.p".format(config['dataset']))
		print(graph_builder.label2id)
		#sys.exit(-1)
		model = load_model(config)
		print(model.doc_embeddings().shape)
		#print(model.input_embeddings().shape)
		label2emb = dict()
		for k in graph_builder.label2id:
		    label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy() 
		else:
		    print('Missing:',k)

		doc_assignment,top_label_assignment = soft_assign_docs(model.doc_embeddings(), label2emb)
		document_phrase_cnt, inverted_index = collect_statistics('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/full.txt')
		docs = [x[0] for x in top_label_assignment[twin][:100]]
		siblings_docs = []
		for key in siblings:
			if key != twin:
				siblings_docs.append([x[0] for x in top_label_assignment[key][:100]])
		phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
		scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index, phrase2idx)
		embed()
	elif config['stage'] == 'test':
	# Find concentrated concepts and specific common sense node
		print("Loading Embedding")
		#Sports Test Documents
		set_docs = [
		[1002, 33719, 62913, 2123, 122759, 36113, 35827, 16109], #korea nuclear
		[1848, 55838, 138468, 55669, 69069, 53809, 23665, 61064, 82084, 61629], #physics
		[5804, 5803, 17361, 20859, 18942, 18336, 21233, 19615, 17945], #basketball
		[51, 256, 381, 169, 45296, 667],
		[52, 357, 629, 936, 801, 1681, 1105, 725],
		[77, 6218, 11847, 615, 1940, 5458, 3169, 10201, 2453, 47171],
		[79, 1163, 2576, 15069, 2836, 11288, 3169, 1680, 14666, 5646, 11569],
		[99, 2323, 14379, 4422, 4573, 5148, 292, 1322, 6811, 6654, 382]
		]

		graph_builder = textGraph(None)
		graph_builder.load_mapping("{}_mapping.txt".format(config['dataset']))
		graph_builder.load_label("{}_label.p".format(config['dataset']))
		print(graph_builder.label2id)
		#sys.exit(-1)
		model = load_model(config)
		print(model.doc_embeddings().shape)
		#print(model.input_embeddings().shape)
		label2emb = dict()
		for k in graph_builder.label2id:
		    label2emb[k] = model.label_embed.weight[graph_builder.label2id[k], :].data.cpu().numpy() 
		else:
		    print('Missing:',k)


		model = load_model(config)
		#doc_emb = load_doc_emb(config['doc_emb_path'])
		#label_emb = load_label_emb(config['label_emb_path'])
		#main_doc_assignment = load_gt_labels('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.json')
		# main_doc_assignment = background_doc_assign(doc_emb, label_emb, [''])
		doc_assignment,top_label_assignment = soft_assign_docs(model.doc_embeddings(), label2emb)
		FILELIST = open("intermediate_data/{}_{}_filelist.txt".format(config['summ_method'], config['dataset']), 'w')
		document_phrase_cnt, inverted_index = collect_statistics('/shared/data/qiz3/text_summ/src/jt_code/doc2cube/tmp_data/full.txt')
		
		for idx,docs in enumerate(set_docs):
			OUT = open("intermediate_data/{}_{}_set{}.txt".format(config['summ_method'], config['dataset'], idx), 'w')
			FILELIST.write("intermediate_data/{}_{}_set{}.txt\n".format(config['summ_method'], config['dataset'], idx))

			#TODO: @jingjing, rewrite target_doc_assign in utils, you can have label2emb.keys instead call concepts
			doc_embeddings = model.doc_embeddings()
			hierarchy = simple_hierarchy()
			label, all_siblings = target_hier_doc_assign(hierarchy, docs, label2emb, doc_embeddings, option='hard')
			print(label, all_siblings)
			# main_label, target_label, sibling_labels = target_doc_assign(concepts, docs, label_emb, doc_emb)
			# main_set = set(map(lambda x:x[0], main_doc_assignment[main_label]))
			# print(len(main_set))
			#siblings = retrieve_siblings(main_set, doc_assignment, sibling_labels, topk=100)
			
			siblings_docs = [map(lambda x:x[0], top_label_assignment[l][:config['topk']]) for l in all_siblings if l != label]
			twin_docs = map(lambda x:x[0], top_label_assignment[label][:config['topk']])
			print(siblings_docs, twin_docs)
			print("Number of sibling groups: {}".format(len(siblings_docs)))

			

			ranked_list = []

			if config['summ_method'] == 'caseOLAP':
				phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
				scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index,
	                                                           phrase2idx)

			elif config['summ_method'] == 'caseOLAP-twin': 

				#############################
				# Diversified ranking block #
				#############################
				'''
				phrase2idx, idx2phrase = build_in_domain_dict(twin_docs, document_phrase_cnt)
				scores, ranked_list = generate_caseOLAP_scores(siblings_docs, twin_docs, document_phrase_cnt, inverted_index, phrase2idx)
				
				twin_rank = list(map(lambda x:x[0], ranked_list))

				similarity_scores, _ = calculate_pairwise_similarity(phrase2idx)
				selected_index = select_phrases(scores, similarity_scores, 2, 1000)
				phrases = [idx2phrase[k] for k in selected_index]
				'''

				##################
				# caseOLAP block #
				##################
				phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
				scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index,
	                                                           phrase2idx)

			elif config['summ_method'] == 'textrank':
				###################
	            # Textrank block ##
	            ###################

				phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
				similarity_scores = build_co_occurrence_matrix(docs, phrase2idx,
				        '/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt')
				scores = textrank(phrase2idx.keys(), similarity_scores)
				ranked_list = [(idx2phrase[i], score) for (i, score) in enumerate(scores)]
				ranked_list = sorted(ranked_list, key=lambda t:-t[1])
			elif config['summ_method'] == 'kams':
				phrase2idx, idx2phrase = build_in_domain_dict(docs, document_phrase_cnt)
				scores, ranked_list = generate_caseOLAP_scores(siblings_docs, docs, document_phrase_cnt, inverted_index, phrase2idx)
				labels = dict()
				for r in ranked_list[:30]:
					labels[r[0]] = 1
				for r in ranked_list[-30:]:
					labels[r[0]] = 0
				similarity_scores = build_co_occurrence_matrix(docs, phrase2idx,
				        '/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt')
				
				ranked_list = GCNRanker(phrase2idx.keys(), similarity_scores, phrase2idx, idx2phrase, labels)
				ranked_list = sorted(ranked_list, key=lambda x:x[1], reverse=True)

				'''
				ranked_lists = []
				for doc in docs:
					phrase2idx, idx2phrase = build_in_domain_dict([doc], document_phrase_cnt)
					similarity_scores = build_co_occurrence_matrix([doc], phrase2idx,
					        '/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/segmentation.txt')
					scores = textrank(phrase2idx.keys(), similarity_scores)
					sub_ranked_list = [(idx2phrase[i], score) for (i, score) in enumerate(scores)]
					sub_ranked_list = sorted(sub_ranked_list, key=lambda t:-t[1])
					ranked_lists.append(sub_ranked_list)
				
				#print(doc, ranked_list[:20], rank_in_twins)
				#print('**'.join(list(map(lambda x:x[0], ranked_list[:10]))))
				#embed()

				ranked_list = ensembleSumm(ranked_lists, k=30)
				'''

				
			OUT.write(' '.join(map(str, docs)) + '\n')
			for r in ranked_list[:30]:
				OUT.write("{} {}\n".format(r[0], r[1]))
			##########################
			# Manifold ranking block #
			##########################
			'''
			phrase2idx, idx2phrase = build_in_domain_dict(twin_docs, document_phrase_cnt)
			scores, ranked_list = generate_caseOLAP_scores(siblings_docs, twin_docs, document_phrase_cnt,
														   inverted_index,
														   phrase2idx)
			phrase_selected = 1000
			all_phrases = [t[0] for t in ranked_list[:phrase_selected]]
			phrase2idx = {phrase: i for (i, phrase) in enumerate(all_phrases)}
			idx2phrase = {phrase2idx[k]: k for k in phrase2idx}
			similarity_scores, _ = calculate_pairwise_similarity(phrase2idx)
			topic_scores = np.zeros([len(phrase2idx)])
			for i in range(phrase_selected):
				topic_scores[phrase2idx[ranked_list[i][0]]] = ranked_list[i][1]
			target_phrase2idx = generate_candidate_phrases(document_phrase_cnt, docs)
			target_phrases = [phrase for phrase in phrase2idx if phrase in target_phrase2idx]
			twin_phrases = [phrase for phrase in phrase2idx if phrase not in target_phrase2idx]
			A = manifold_ranking(twin_phrases, target_phrases, topic_scores, phrase2idx, similarity_scores)
			ranked_list = [(phrase, A[phrase2idx[phrase]]) for phrase in target_phrases]
			ranked_list = sorted(ranked_list, key=lambda t: -t[1])
			'''
			break
	elif config['stage'] == 'finetune':
		pass
