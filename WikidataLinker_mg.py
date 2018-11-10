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

relation_dic=defaultdict(int)
class WikidataLinker:
	def __init__(self,relation_list=None):
		self.tid2title=defaultdict(str)
		#self.title2tid=defaultdict(str)
		self.client = MongoClient('mongodb://dmserv4.cs.illinois.edu:27017')
		self.db=self.client.wikidata
		self.property=defaultdict(int)
		self.kb={}
		self.relation_list = relation_list
		self.p_list=[]

	def wid2title(self,wid):
		page = self.db.enwiki_pages.find_one({'id': wid})
		title=''
		#print(page)
		if page is not None:
			title =page['sitelinks']['enwiki']['title']
		return title   

	def title2wid(self,title):
		wid=''
		page = self.db.enwiki_pages.find_one({'sitelinks.enwiki.title': title})
		if page is not None:
			wid=page['id']
		return wid

	def get_adj(self,wid):
		object=self.db.enwiki_pages.find_one({'id': wid})
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
							if not self.relation_list or p in self.relation_list:
								self.kb[(wid,obj_wid)] = self.relation_list.index(p) + 1
						adj_nodes.add(obj_wid)
		return adj_nodes

	def retrieve_facts(self, kws, total_hop):
		candidates = defaultdict(int)
		current_hop=1
		new_kws={}
		temp_kws=list(kws.keys())
		while current_hop <=total_hop:
			for kw in temp_kws:
				#print(kw)
				adj_nodes=self.get_adj(kw)
				#print(adj_nodes)
				for node in adj_nodes:
					if node not in kws:
						new_kws[node]=1    
						candidates[node] += 1
			#print(candidates)
			for c in candidates:
				# condition
				if candidates[c] >= 2 and c not in kws:
					kws[c]=1
			temp_kws=list(new_kws.keys())
			new_kws={} 
			candidates = defaultdict(int)
			#print('hop ',current_hop,'done.') 
			current_hop+=1
		#for kw in kws:
			#print(self.wid2title(kw),)
		facts = []
		for i,k in enumerate(list(kws.keys())):
			for j,l in enumerate(list(kws.keys())):
				if k == l:
					continue
				if (k,l) in self.kb:
					p=self.kb[(k,l)]
					self.property[p]+=1	
					k_title=self.wid2title(k)
					l_title=self.wid2title(l)
					if k_title =='' or l_title=='':
						continue
					facts.append([k_title.replace(' ','_').lower(), l_title.replace(' ','_').lower(), p])
		return facts

	def expand(self,doc_list,num_hop):
		phrases=self.load_kws(doc_list)
		kws={}
		for phrase in phrases:
			wid=self.title2wid(phrase)
			if  wid !='':
				kws[wid]=1
		return self.retrieve_facts(kws,num_hop)
		
		#for p,count in self.property.items():
		#	self.p_list.append((p,count))
		

	def output_stats(self):
		self.p_list.sort(key=itemgetter(1),reverse=True)
		for tuple in self.p_list:
			print(tuple[0],tuple[1])

	def load_kws(self,input_keywords):
		input_kws = []
		for tp in input_keywords:
			word=tp[0]
			ner_type=tp[3]
			if ner_type =='TIME' or ner_type=='DATE' or ner_type=='CARDINAL' or ner_type=='ORDINAL' or ner_type=='GPE':
				continue
			if word not in input_kws:
				input_kws.append(word)
		return input_kws
   

if __name__ == '__main__': 
	tmp=WikidataLinker()
	#phrases=tmp.load_kws([31])
	#print(phrases)
	tmp.expand([0,1,2,3,4],1)
	embed()
	#exit()
