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
config = {'batch_size': 128, 'epoch_number': 50, 'emb_size': 100, 'kb_emb_size': 100, 'num_sample': 5, 'gpu':0,
		'model_dir':'/shared/data/qiz3/text_summ/src/model/', 'dataset':'NYT_sports', 'method':'knowledge2vec',
		'preprocess': True}

#P54 team P31 instance of P27 nationality P641 sports P413 position
#P106 occupation P1344 participant P17 country P69 educate P279 subclass of P463 member of P641 sport

if config['preprocess']:
	print("Extracting Hierarchies")
	tmp = WikidataLinker(relation_list)
	graph_builder = textGraph(tmp)
	graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
	#graph_builder._load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.txt')
	h, t2wid, wid2surface = graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.token', '/shared/data/qiz3/text_summ/data/NYT_sports.json', attn=True)

	# Example usage of creating concept from hierarchies
	concepts = []
	concept_configs = [[(u'type of sport', 2), [2,4]]]
	for con_config in concept_configs:
		concept = Concept(h)
		concept.construct_concepts(con_config)
		concepts.append(concept)

	graph_builder.load_concepts(concepts)
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

# concept.concept_link([phrases]), return height=1 non-leaf node linked in the concepts
# pickle.dump(concept, open('sports.concept', 'wb'))
# embed()


#Interface of using various embeddings
embedder.Train(config, X, y, num_words, num_docs)



# Assign documents on specific node(hard), very basic baseline
assignments = background_assign(concept, t2wid, wid2surface)
embed()