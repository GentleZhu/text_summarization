from collections import defaultdict
from multiprocessing import Pool
import argparse
import re
import itertools
from scipy.sparse import coo_matrix
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from operator import itemgetter
import json
from IPython import embed
from tqdm import tqdm
from copy import deepcopy

relation_dic=defaultdict(int)
class WikidataLinker:
	def __init__(self,relation_list1=None, relation_list2=None):
		self.tid2title=defaultdict(str)
		#self.title2tid=defaultdict(str)
		self.client = MongoClient('mongodb://dmserv4.cs.illinois.edu:27018')
		self.db=self.client.wikidata
		self.property=defaultdict(int)
		self.kb={}
		self.relation_list1 = relation_list1
		self.relation_list2 = relation_list2
		self.p_list=[]

	def wid2title(self,wid):
		page = self.db.pages.find_one({'id': wid})
		title=''
		#print(page)
		if page is not None:
			#title =page['sitelinks']['enwiki']['title']
			try:
				title = page['labels']['en']['value']
			except:
				pass
		return title   

	# bug need to be fixed
	def title2wid(self,title):
		wid=''
		page = self.db.pages.find_one({'sitelinks.enwiki.title': title})
		if page is not None:
			wid=page['id']
		return wid

	def get_adj(self,wid, hop):
		object=self.db.pages.find_one({'id': wid})
		adj_nodes=set()
		self.kb[wid]={}
		#print(wid)
		for p in object['claims']:
			for item in object['claims'][p]:
				try:
					batch=item['mainsnak']['datavalue']['value']
				except KeyError:
					continue
				if 'id' in batch and isinstance(batch, dict):
					obj_wid=batch['id']
					if obj_wid not in adj_nodes and self.wid2title(obj_wid)!='':
						#print(obj_wid,self.wid2title(obj_wid))
						if (wid,obj_wid) not in self.kb:
							if hop == 1:
								relation_list = self.relation_list1
							else:
								relation_list = self.relation_list2
							if not relation_list or p in relation_list:
								if not relation_list:
									self.kb[(wid, obj_wid)] = (0, hop)
								else:
									self.kb[(wid,obj_wid)] = (p, hop)
						if (wid, obj_wid) in self.kb:
							adj_nodes.add(obj_wid)
		return adj_nodes

	def retrieve_facts(self, kws, total_hop):
		current_hop=1
		temp_kws=deepcopy(kws)
		while current_hop <=total_hop:
			for kw in tqdm(temp_kws):
				if temp_kws[kw] < current_hop:
					continue
				adj_nodes=self.get_adj(kw, current_hop)
				for node in adj_nodes:
					if True:#node not in kws:
						kws[node] = current_hop + 1

			temp_kws=deepcopy(kws)
			current_hop += 1

		facts = []
		t2wid = {}
		for i,k in enumerate(list(kws.keys())):
			for j,l in enumerate(list(kws.keys())):
				if k == l:
					continue
				if (k,l) in self.kb:
					p=self.kb[(k,l)]
					self.property[p]+=1	
					k_title=self.wid2title(k)
					l_title=self.wid2title(l)
					t2wid[k_title] = k
					t2wid[l_title] = l
					if k_title =='' or l_title=='' or l == 'Q4167410':
						continue
					#facts.append([k_title.replace(' ','_').lower(), l_title.replace(' ','_').lower(), p])
					facts.append([k_title, l_title, p[0], p[1]])
		return facts, t2wid

	def expand(self,doc_list,num_hop):
		phrases=self.load_kws(doc_list)
		kws={}
		#print(phrases)
		wid2original = defaultdict(list)
		print('title2wid...')
		for phrase in tqdm(phrases):
			wid=self.title2wid(phrase.replace('_', ' '))
			if  wid !='':
				kws[wid]=1
				wid2original[wid].append(phrase)
		facts, t2wid = self.retrieve_facts(kws,num_hop)
		return facts, t2wid, wid2original
		
		#for p,count in self.property.items():
		#	self.p_list.append((p,count))
		

	def output_stats(self):
		self.p_list.sort(key=itemgetter(1),reverse=True)
		for tuple in self.p_list:
			print(tuple[0],tuple[1])

	def load_kws(self,input_keywords):
		input_kws = set()
		print('load kws...')
		for tp in tqdm(input_keywords):
			word=tp
			if False:
				ner_type=tp[3]
				if ner_type in ['TIME', 'DATE', 'CARDINAL', 'ORDINAL']:
					continue
			#if word not in input_kws:
			#	input_kws.append(word)
			input_kws.add(word)
		return list(input_kws)
   

if __name__ == '__main__': 
	tmp=WikidataLinker()
	#phrases=tmp.load_kws([31])
	#print(phrases)
	tmp.expand([0,1,2,3,4],1)
	embed()
	#exit()
