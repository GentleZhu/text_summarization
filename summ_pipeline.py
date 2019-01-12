import sys
from utils.WikidataLinker_mg import WikidataLinker
from tqdm import tqdm
from utils.utils import textGraph, Concept, background_assign
import pickle
from phraseExtractor import phraseExtractor
sys.path.append('./models/')
import embedder
from IPython import embed

relation_list=['P54', 'P31', 'P27', 'P641', 'P413', 'P106', 'P1344', 'P17', 'P69', 'P279', 'P463', 'P641']
#relation_list=['P31', 'P641']
config = {'batch_size': 128, 'epoch_number': 101, 'emb_size': 100, 'kb_emb_size': 100, 'num_sample': 5, 'gpu':0,
		'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'NYT_sports', 'method':'knowledge2skip_gram', 'id':3,
		'preprocess': True, 'relation_list':[]}

#P54 team P31 instance of P27 nationality P641 sports P413 position
#P106 occupation P1344 participant P17 country P69 educate P279 subclass of P463 member of P641 sport


#Interface of using various embeddings
if config['stage'] == 'train':
	if config['preprocess']:
		
		tmp = WikidataLinker(relation_list)
		graph_builder = textGraph(tmp)
		graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
		#graph_builder._load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.txt')
		print("Extracting Hierarchies")
		if len(config['relation_list']) > 0:
			#h, t2wid, wid2surface = graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.token', '/shared/data/qiz3/text_summ/data/NYT_sports.json', attn=True)
			#h.save_hierarchy("{}_{}_hierarchies.p".format(config['method'], config['dataset']))
			h = pickle.load(open("{}_{}_hierarchies.p".format(config['method'], config['dataset']), 'rb'))
			concepts = []
			concept_configs = [[(u'type of sport', 2), [2, 4]]]
			for con_config in concept_configs:
				concept = Concept(h)
				concept.construct_concepts(con_config)
				concepts.append(concept)
				concept.output_concepts('concepts.txt')
			graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.token', '/shared/data/qiz3/text_summ/data/NYT_sports.json', attn=False)
			graph_builder.load_concepts('/shared/data/qiz3/text_summ/data/NYT_sports.token', concepts)
			#sys.exit(-1)
		else:
			h = pickle.load(open("{}_{}_hierarchies.p".format(config['method'], config['dataset']), 'rb'))
			# Example usage of creating concept from hierarchies
			concepts = []
			# TODO(QI): make it more distributional
			concept_configs = [[(u'type of sport', 2), [2, 4]]]
			concept = Concept(h)
			concept.root = (u'type of sport', 2)
			concept.links = {
				'soccer': [['type_of_sport|soccer', 0]],
				'basketball': [['type_of_sport|basketball', 0]],
				'baseball': [['type_of_sport|baseball', 0]],
				'football': [['type_of_sport|football', 0]],
				'tennis': [['type_of_sport|tennis', 0]],
				'golf': [['type_of_sport|golf', 0]],
				'hockey': [['type_of_sport|hockey', 0]]
			}
			concepts.append(concept)
			graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.token', '/shared/data/qiz3/text_summ/data/NYT_sports.json', attn=False)
			graph_builder.load_concepts('/shared/data/qiz3/text_summ/data/NYT_sports.token', concepts)
		graph_builder.normalize_text()
		graph_builder.build_dictionary()
		graph_builder.text_to_numbers()

		X, y, num_docs, num_words = graph_builder.buildTrain(method=config['method'])
		save_point = {'X': X, 'y':y, 'num_docs':num_docs, 'num_words':num_words}
		graph_builder.dump_mapping("{}_mapping.txt".format(config['dataset']))
		pickle.dump(save_point, open("{}_{}.p".format(config['method'], config['dataset']), 'wb'))
	else:
		save_point = pickle.load(open("{}_{}.p".format(config['method'], config['dataset']), 'rb'))
		X = save_point['X']
		y = save_point['y']
		num_docs = save_point['num_docs']
		num_words = save_point['num_words']
	
	print("Training Embedding")
	embedder.Train(config, X, y, num_words, num_docs)
else:
	print("Loading Embedding")

# Find concentrated concepts and 
#assignments = background_assign(concept, t2wid, wid2surface)
#embed()