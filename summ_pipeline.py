#Preprocessing
import sys
#from FreeBaseLinker import FreebaseLinker
from utils.WikidataLinker_mg import WikidataLinker
from tqdm import tqdm
from utils.utils import textGraph
import pickle

#relation_list=['people.person.nationality','people.person.profession','location.location.contains']
relation_list=['P54', 'P31', 'P27', 'P641', 'P413', 'P106', 'P1344', 'P17', 'P69', 'P279', 'P463', 'P641']
#relation_list=['P31', 'P17', 'P106', 'P279']
#tmp = FreebaseLinker(relation_list)

#P54 team P31 instance of P27 nationality P641 sports P413 position
#P106 occupation P1344 participant P17 country P69 educate P279 subclass of P463 member of P641 sport

### Comment below lines 

print("Extracting Hierarchies")

tmp = WikidataLinker(relation_list)
graph_builder = textGraph(tmp)
graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
#graph_builder._load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.txt')
h = graph_builder.load_corpus('/shared/data/qiz3/text_summ/data/NYT_sports.token', '/shared/data/qiz3/text_summ/data/NYT_sports.json', attn=True)

# Example usage of create concept from hierarchies
concept = Concept(h)
concept.construct_concepts([(u'type of sport', 2), [2,4]])
# concept.concept_link([phrases]), return height=1 non-leaf node linked in the concepts