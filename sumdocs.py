import sys, subprocess, re
import time
from tqdm import tqdm
from utils.utils import *
from utils.metrics import compute_rouge_l
from baselines.baselines import densitypeak,graphdegen
import pickle #, torch
from scipy import spatial
from collections import defaultdict, Counter
#from phraseExtractor import phraseExtractor
sys.path.append('./models/')
from summarizer import collect_statistics, build_background_dict, generate_caseOLAP_scores, _generate_caseOLAP_scores, build_co_occurrence_matrix
import summarizer
import configparser
import numpy
from IPython import embed
from baselines.baselines import *
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path

from nltk.tokenize import sent_tokenize
from multiprocessing import Pool
#from wmd import WMD

def calc_diversity(stringID, stringBank, cur, sim):
    if len(cur) == 0:
        return 0
    
    for i in cur:
        if (stringID, i) not in sim:
            sim[(stringID, i)] = compute_rouge_l(stringBank[stringID], stringBank[i], mode='p')
    score = -max([sim[(stringID, i)] for i in cur])
    return score

def calc_diversity_v2(stringID, stringBank, cur, sim, phrase_scores):
    if len(cur) == 0:
        return 0
    cnt = 0
    tokens = set()
    for i in cur:
        tokens |= set(stringBank[i])
    tokens |= set(stringBank[stringID])
        #if (stringID, i) not in sim:
        #sim[(stringID, i)] = compute_rouge_l(stringBank[stringID], stringBank[i], mode='p')
    # score = -max([sim[(stringID, i)] for i in cur])
    # embed()
    return len(tokens & phrase_scores.keys()) / len(phrase_scores)

def calc_score(stringID, stringBank, pick, cur, dis, option = 'rouge', phrase_scores = None):
    # cur is current pick
    if stringID in cur:
        return -1
    if option == 'rouge':
        I = dis[stringID]
        #if stringID in dis:
        # I = compute_rouge_n([stringID], stringB, n=2)
    D = calc_diversity_v2(stringID, stringBank, pick, None, phrase_scores)
    
    alpha = 0.8
    return I + 2 * D
    #return D
import random
# why duplication happens
def mmr_selector(doc_set, phrase_scores, doc_id = None, OUT = None, limits = 100):
    #with open(fileOut, 'w+') as OUT:
        # a,b = IN1.readlines(), IN2.readlines()
    sim = defaultdict(float)
    dis = []
    cur = []
    pick = []
    length = 0
    # adjusting the phrase scores
    for sentence in doc_set:
        dis.append(sum([phrase_scores[phrase] if phrase in phrase_scores else 0 for phrase in list(set(sentence)) ]) / len(sentence))

    t1 = time.time()
    prev_F = 0.0
    current_ = 0.0
    # 
    #
    if False:
        ranks = list(range(len(doc_set)))
        random.shuffle(ranks)
    idx = 0

    # here is a bug for it
    while True:
        scores = []
        if True:
            for idx in range(len(doc_set)):
                #if idx in cur:
                #    continue
                #scores = dis[idx] + calc_score(idx, doc_set, pick, sim, dis, 'rouge', phrase_scores)
                scores.append(calc_score(idx, doc_set, pick, cur, dis, 'rouge', phrase_scores))
            # embed()
            if max(scores) < 0:
                break
            
            cur.append(np.argmax(scores))
        else:
            cur.append(ranks[idx])
            idx += 1
        
        
        if length + len(doc_set[cur[-1]]) <= limits:
            length += len(doc_set[cur[-1]])
            current_ += dis[cur[-1]]
            pick.append(cur[-1])
        if len(cur) == len(doc_set):
            break
    t2 = time.time()
    if len(pick) != len(set(pick)):
        embed()
    out_text = []
    for idx in pick:
        out_text.append(' '.join(doc_set[idx]))
    if OUT is None:
        return ' '.join(out_text)
    else:
        if doc_ids is None:
            OUT.write('{}\n'.format(' '.join(out_text)).replace('_', ' ') )
        else:
            #OUT.write('{}\t{}\n'.format(doc_id, ' '.join(out_text)) )
            OUT.write('{}\n'.format(' '.join(out_text)).replace('_', ' ') )
        OUT.flush()

def calculate_similarity(passages):
    sent_num = len(passages)
    similarity_scores = np.zeros([sent_num, sent_num])
    for i in range(sent_num):
        for j in range(sent_num):
            if j < i:
                similarity_scores[i][j] = similarity_scores[j][i]
            else:
                #sim = 0
                #for word in passages[i]:
                #    if word in passages[j]:
                #        sim += 1
                #sim /= (math.log(len(set(passages[i])) + 1) + math.log(len(set(passages[j])) + 1))
                sim = len(set(passages[i]) & set(passages[j])) / (math.log(len(set(passages[j])) + 1) + math.log(len(set(passages[j])) + 1))
                similarity_scores[i][j] = sim
    return similarity_scores

def textrank(doc_set, OUT, limits = 100):
    alpha = 0.85
    threshold = 0.0001
    # passages, raw_sentences = generate_duc_docs(s, stopword_path)
    similarity_scores = calculate_similarity(doc_set)
    normalized_sim = np.zeros(similarity_scores.shape)
    for i in range(similarity_scores.shape[0]):
        similarity_scores[i, i] = 0.0
        sum_ = np.sum(similarity_scores[:, i])
        if sum_ == 0:
            continue
        for j in range(similarity_scores.shape[0]):
            normalized_sim[j, i] = similarity_scores[j, i] / sum_

    sent_num = len(doc_set)
    scores = 1.0 / sent_num * np.ones([sent_num])
    current_scores = scores.copy()

    while True:
        #print('Update...')
        scores = alpha * np.dot(normalized_sim, scores) + (1 - alpha)
        dist = np.linalg.norm(current_scores - scores)
        if dist < threshold:
            break
        current_scores = scores

    ranked_list = []
    for idx, score in enumerate(scores):
        ranked_list.append((idx, scores[idx]))
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    # print(ranked_list)
    length = 0

    out_text = []
    for r in ranked_list:
        
        
        # if length > 60:
        # TODO: add a paramter for maximum length of summarization
        if length +  len(doc_set[r[0]])> limits:
            continue
        length += len(doc_set[r[0]])
        out_text.append(' '.join(doc_set[r[0]]).replace('_', ' '))
    #print('doc_set is', doc_set )
    #print('####################################')
    #print('OUT is', out_text)
    OUT.write(' '.join(out_text) + '\n')
    OUT.flush()

def lexrank(lxr,sentences,OUT,limits=100):
    #out_text=lxr.get_ranked_sentences(sentences)
    all_summary = lxr.get_summary(sentences, summary_size = len(sentences))
    length = 0
    out_text = []
    for s in all_summary:
        if length + len(s.split()) > limits:
            break
        length += len(s.split())
        out_text.append(s.replace('_', ' '))
    OUT.write(' '.join(out_text) + '\n')
    OUT.flush()

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
    # 1010
    config['num_siblings'] = int(config['num_siblings'])
    config['input_file'] = config['input_file']
    config['output_file'] = config['output_file']
    config['preprocess'] = json.loads(config['preprocess'].lower())
    config['next_sent'] = config['next_sent']
    # config['next_sent'] = '\t'
    config['word_limits'] = int(config['word_limits'])
    config['split_idx']=int(config['split_idx'])
    return config
    

# add topics.txt and num_seeds, it will replace the expanded words with topic in topcate.mix_emb_topics_t.txt
def load_cate_new(config):
    cate_path=config['cate_path']
    w_emb_file=cate_path+'topcate.mix_emb_topics_w.txt'
    t_emb_file=cate_path+'topcate.mix_emb_topics_t.txt'
    d_emb_file=cate_path+'topcate.mix_emb_topics_d.txt'
    split_idx=config['split_idx']
    num_seeds=int(config['num_seeds'])
    topics_file=cate_path+'topics.txt'
    word_embs=[]
    word2id=dict()
    i=0
    index=0
    with open(w_emb_file,'rb') as IN:
    #for i,line in enumerate(w_file):
        for line in IN:
            if i==0:
                i+=1
                continue
            try:
                line=line.decode("utf-8")
            except:
            #print('encoding error is ',line)
                continue
            line=line.strip().split(' ')

            if len(line)!=101:
                continue
            word=line[0]
            embedding=[float(x) for x in line[1:]]
            if len(embedding)==100:
                word_embs.append(embedding)
                word2id[word]=index
                index+=1

    label2emb=dict()
    i=0
    cate_list=[]
    with open(t_emb_file,'rb') as IN1, open(topics_file) as IN2:
        topics=[]
        for line in IN2:
            topics.append(line.strip())
        for line in IN1:
            if i==0:
                i+=1
                continue
            try:
                line=line.decode("utf-8")
            except:
                #print('encoding error is ',line)
                continue
            line=line.strip().split(' ')
            
            label=topics[i-1]
            #print(label)
            i+=1
            embedding=[float(x) for x in line[num_seeds:]]
            #print(len(embedding))
            if len(embedding)==100:
                label2emb[label]=embedding

    doc2emb=[]
    i=0
    with open(d_emb_file,'rb') as IN:
        for line in IN:
            if i==0:
                i+=1
                continue
            try:
                line=line.decode("utf-8")
            except:
            #print('encoding error is ',line)
                continue
            line=line.strip().split(' ')
            if len(line)!=101:
                continue
            doc_id=line[0]
            embedding=[float(x) for x in line[1:]]
            if len(embedding)==100:
                doc2emb.append(embedding)
    doc2emb=numpy.array(doc2emb, dtype=np.float32)
    word_embs=numpy.array(word_embs, dtype=np.float32)
    train2emb=doc2emb[0:split_idx]
    test2emb=doc2emb[split_idx:]
    return word_embs,word2id,label2emb,train2emb,test2emb,topics

#add cate_path to config file
#word_embs,word2id,label2emb,doc2emb= load_cate(config)
def load_cate(config):
    cate_path=config['cate_path']
    w_emb_file=cate_path+'emb_topics_w.txt'
    t_emb_file=cate_path+'emb_topics_t.txt'
    d_emb_file=cate_path+'emb_topics_d.txt'
    topics_file=cate_path+'topics.txt'
    split_idx=config['split_idx']
    word_embs=[]
    word2id=dict()
    i=0
    index=0
    with open(w_emb_file,'rb') as IN, open(topics_file) as IN2:
        topics=[]
        for line in IN2:
            topics.append(line.strip())
    #for i,line in enumerate(w_file):
        for line in IN:
            if i==0:
                i+=1
                continue
            try:
                line=line.decode("utf-8")
            except:
            #print('encoding error is ',line)
                continue
            line=line.strip().split(' ')

            if len(line)!=101:
                continue
            word=line[0]
            embedding=[float(x) for x in line[1:]]
            if len(embedding)==100:
                word_embs.append(embedding)
                word2id[word]=index
                index+=1

    label2emb=dict()
    i=0
    with open(t_emb_file,'rb') as IN:
        for line in IN:
            if i==0:
                i+=1
                continue
            try:
                line=line.decode("utf-8")
            except:
            #print('encoding error is ',line)
                continue
            line=line.strip().split(' ')
            if len(line)!=101:
                continue
            label=line[0]
            embedding=[float(x) for x in line[1:]]
            if len(embedding)==100:
                label2emb[label]=embedding

    doc2emb=[]
    i=0
    with open(d_emb_file,'rb') as IN:
        for line in IN:
            if i==0:
                i+=1
                continue
            try:
                line=line.decode("utf-8")
            except:
            #print('encoding error is ',line)
                continue
            line=line.strip().split(' ')
            if len(line)!=101:
                continue
            doc_id=line[0]
            embedding=[float(x) for x in line[1:]]
            if len(embedding)==100:
                doc2emb.append(embedding)
    doc2emb=numpy.array(doc2emb, dtype=np.float32)
    word_embs=numpy.array(word_embs, dtype=np.float32)
    train2emb=doc2emb[0:split_idx]
    test2emb=doc2emb[split_idx:]
    # embed()
    return word_embs,word2id,label2emb,train2emb,test2emb,topics

def sumdocs(docs, tokenized_sents, offset, line_cnt, doc_id):
    #docs, tokenized_sents, offset, line_cnt, doc_id = args[0], args[1], args[2], args[3], args[4]
    global config, train2emb, test2emb, top_label_assignment, topics, document_phrase_cnt, inverted_index, OUT, comparative_dict, graph_builder

    start_time = time.time()
    if config['summ_method'] == 'sumdocs':
        # default is KNN search for CATE embedding
        if config['comparative_opt'] == 'knn': #KNN comparative search, route 0 and route 1
            count = defaultdict(int)
            for doc_agg_emb in test2emb[offset:offset+line_cnt]:
                sim_max = -1
                category = None
                for label in label2emb:
                    sim = 1 - spatial.distance.cosine(doc_agg_emb, label2emb[label])
                    if sim > sim_max:
                        sim_max = sim
                        category = label
                count[category]+= 1
            #print(count)
            
            category = max(count.items(), key=operator.itemgetter(1))[0]
            comp_pool = list(map(lambda x: x[0],top_label_assignment[category]))
            #all_siblings = ['food', 'drinks', 'ambience', 'service', 'price']
            #all_siblings = ['science', 'politics', 'business', 'sport']
            all_siblings = topics

            # changes: 1010
            twin_docs = list(map(lambda x: x[0], top_label_assignment[category][:config['num_siblings']]))
            siblings_docs = [list(map(lambda x: x[0], top_label_assignment[l][:config['num_siblings']])) for l in all_siblings
                                 if l != category]
            # 
            t1 = time.time()
            comparative_docs = summarizer.compare(config, None, None, None, test2emb[offset:offset+line_cnt], train2emb, skip_doc=None, contain_doc=comp_pool)
            
            #print(time.time() - t1)
            t1 = time.time()
            phrase_scores = summarizer.summary(config, docs, siblings_docs, twin_docs, comparative_docs, document_phrase_cnt, inverted_index, graph_builder=graph_builder)
            #print(time.time() - t1)
    # @fang
    elif config['summ_method'] == 'sumdocs_wo_twins':
        count = defaultdict(int)
        for doc_agg_emb in test2emb[offset:offset+line_cnt]:
            sim_max = -1
            category = None
            for label in label2emb:
                sim = 1 - spatial.distance.cosine(doc_agg_emb, label2emb[label])
                if sim > sim_max:
                    sim_max = sim
                    category = label
            count[category]+= 1
        #print(count)
        
        category = max(count.items(), key=operator.itemgetter(1))[0]
        comp_pool = list(map(lambda x: x[0],top_label_assignment[category]))
        #all_siblings = ['food', 'drinks', 'ambience', 'service', 'price']
        #all_siblings = ['science', 'politics', 'business', 'sport']
        all_siblings = topics

        # changes: 1010
        twin_docs = list(map(lambda x: x[0], top_label_assignment[category][:config['num_siblings']]))
        siblings_docs = [list(map(lambda x: x[0], top_label_assignment[l][:config['num_siblings']])) for l in all_siblings
                                if l != category]
        category, comparative_docs = '', []
        phrase_scores = summarizer.summary(config, docs, siblings_docs, twin_docs, None, document_phrase_cnt, inverted_index, graph_builder=graph_builder)
    elif config['summ_method'] == 'sumdocs_textrank':
        category, comparative_docs = '', []
        phrase_scores = summarizer.summary(config, docs, None, None, None, document_phrase_cnt, inverted_index, graph_builder=graph_builder)
    elif config['summ_method'] == 'graph_degen':
        category, comparative_docs = '', []
        phrase_scores = graphdegen(docs)
    else:
        assert True == False
    #
    #print(time.time() - start_time)
    #start_time = time.time()

    mmr_selector(tokenized_sents, phrase_scores, doc_id, OUT, limits = config['word_limits'])
    return category, comparative_docs
    #return category, comparative_docs
    return

#def graph_degen(tokenized_sents, document_phrase_cnt)

'''
test.src: each review set
test.tgt: each review set
cate embedding: split_index, before each review in train, after each review test()
'''

if __name__ == '__main__':

    config_file = sys.argv[1]
    config = load_config(config_file)
    # Find concentrated concepts and specific common sense node
    graph_builder = textGraph(None)
    graph_builder.load_stopwords('/shared/data/qiz3/text_summ/data/stopwords.txt')
    
    # Enable if parallel is required to speed up
    # pre-compute the comparative documents
    if False:
        comparative_dict = {}
        with open('new_result/JanResult/multi-news.test.full.sumdocs.txt.temp') as IN_dict:
            for line in IN_dict:
                tmp = line.strip().split('\t')
                comparative_dict[int(tmp[0])] = list(map(int, tmp[1].split()))

    with open(config['input_file']) as IN, open(config['output_file'], 'w') as OUT:
        
        print("Input file is {}".format(config['input_file']))
        if config['method'] == 'Cate':
            document_phrase_cnt, inverted_index = collect_statistics(
                        # '/shared/data/qiz3/text_summ/text_summarization/dataset/{}/train.phraselist.src'.format(config['dataset']))
                        config['phrase_list'])
            # pre-load the background corpus
            # word_embs, word2id, label2emb, train2emb, test2emb, topics = load_cate_new(config)
            word_embs, word2id, label2emb, train2emb, test2emb, topics = load_cate(config)

            bkg_corpus, id_bkg_corpus = [],[]
            with open(config['background_file']) as BKG:
                doc_dict = {}
                doc_id = 0
                for line in BKG:
                    bkg_corpus.append(line.strip())
                    
                    '''
                    id_bkg_corpus = [word2id[x] for x in line.split() if x in word2id]
                    id_list, d_weights = [], np.array([], dtype=np.float32)
                    counts = Counter(id_bkg_corpus)
                    for k in counts.keys():
                        id_list.append(k)
                        d_weights = np.append(d_weights, counts[k])
                    doc_dict[doc_id] = (doc_id, id_list, d_weights)
                    doc_id += 1
                    if doc_id % 10000 == 0:
                        break
                    '''
                    #if doc_id == 10:
                    #    break
                #calc = WMD(word_embs, doc_dict, vocabulary_min=1)
                #print(doc_dict)
                #print(calc.nearest_neighbors(0))

            # embed()
        #print(train2emb.shape,test2emb.shape)
        # embed()
        doc_assignment, top_label_assignment = soft_assign_docs(train2emb, label2emb)
        
        offset = 0
        #if False:
        if config['summ_method'] in ['lexrank', 'densitypeak']:
            documents=[]
            raw_sentences=[]
            stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
            stopwords = set()
            with open(stopword_path) as IN:
                for line in IN:
                    stopwords.add(line.strip())
            with open(config['input_file']) as docs_IN:
                #count=0
                passages = []
                for line_num, line in enumerate(docs_IN.readlines()):
                    if line_num <= 309:
                        continue
                    passage = []
                    raw_sentences = sent_tokenize(line.replace(config['next_sent'], ''))
                    documents.append(raw_sentences)
                    e_s = []
                    for sentence in raw_sentences:
                        words = word_tokenize(sentence)
                        for word in words:
                            if not word in stopwords:
                                e_s.append(word.lower())
                        passage.append(e_s)
                    passages.append(passage)
                    
                    #raw_sentences.append(line.replace(config['next_sent'], ''))
            #embed()
            print('Finished building documents.')
            if config['summ_method'] == 'densitypeak':
                for doc, sentences in zip(passages, documents):
                    densitypeak(doc, sentences, OUT, limits=config['word_limits'])
            elif config['summ_method'] == 'graphdegen':
                    graphdegen(tokenized_sents,OUT)
            elif config['summ_method'] == 'lexrank':
                lxr = LexRank(documents, stopwords=STOPWORDS['en'])
                for doc in documents:
                    lexrank(lxr,doc,OUT,limits=config['word_limits'])
        else:
            # tokenization 
            total_offset = 0
            offsets, line_cnts, tokenized_sents_set, doc_sets = [], [], [], []
            import pickle
            idx_list = pickle.load(open('idx_eval_400.p', 'rb'))
            #
            idx_trunk = []
            # embed()
            for doc_id, doc_set in enumerate(IN.readlines()):
                docs = []
                doc_set = doc_set.strip().lstrip('b"').rstrip('"')
                
                tokenized_sents = sent_tokenize(doc_set.replace(config['next_sent'], ''))
                for idx in range(len(tokenized_sents)):
                    tokenized_sents[idx] = tokenized_sents[idx].strip().split(' ')

                line_cnt = 0
                for line in doc_set.split(config['next_sent']):
                    docs.append(graph_builder.normalize(line))
                    idx_trunk.append(idx_list[doc_id] + line_cnt)
                    line_cnt += 1
                
                offsets.append(total_offset)
                line_cnts.append(line_cnt)
                tokenized_sents_set.append(tokenized_sents)
                doc_sets.append(docs)

                total_offset += line_cnt

            doc_ids = range(len(doc_sets))
            test2emb = test2emb[idx_trunk, :]
            cnt = 0
            for a,b,c,d,e in zip(doc_sets, tokenized_sents_set, offsets, line_cnts, doc_ids):
                if config['summ_method'] in ['sumdocs', 'sumdocs_textrank', 'sumdocs_wo_twins', 'graph_degen']:
                    category, twins = sumdocs(a,b,c,d,e)
                elif config['summ_method'] == 'textrank':
                    textrank(b, OUT, limits = config['word_limits'])
                cnt += 1
                OUT.flush()
            print("total count:", cnt)