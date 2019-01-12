import xml.dom.minidom
from collections import defaultdict
from tqdm import tqdm
import sys, json

ct = 0
type_cnt = defaultdict(int)

keys = [('Top/News/World', 116009), ('Top/News/U.S.', 110422), ('Top/News/New York and Region', 73217), ('Top/News/Sports', 61253), ('Top/News/Business', 51994), ('Top/News/Washington', 46996), ('Top/News/Health', 41990), ('Top/News/Technology', 15250), ('Top/News/Corrections', 11114), ('Top/News/Science', 8425), ('Top/News/Education', 7593)]
keys = set(map(lambda x:x[0], keys))
with open("/shared/data/yuningm2/datasets/NYT_annotated_corpus/data/filelist_2003-07.txt") as f, open('../data/NYT_annotated_corpus/NYT_corpus.json', 'w') as OUT_JSON:
    for idx,line in enumerate(tqdm(f.readlines())):
        xml_path = "/shared/data/yuningm2/datasets/NYT_annotated_corpus/data/accum2003-07/" + line.strip()
        #with open(xml_path) as f2:
            #print(f2.readlines())
        dom = xml.dom.minidom.parse(xml_path)
        root = dom.documentElement
        
        #article has abstract
        tags = root.getElementsByTagName('abstract')
        if len(tags) > 0:
            _corpus = {'id':None, 'abstract':[], 'title':None, 'type':[], 'content':[], 'lead_3':[]}
            for tag in tags:
                try:
                    data = tag.childNodes[1].childNodes[0].data
                    _corpus['abstract'].append(data) 
                except:
                    pass

            tags = root.getElementsByTagName('classifier')
            _id = root.getElementsByTagName('doc-id')[0].getAttribute('id-string')
            #desc = defaultdict(set)
            tag_names = []
            for tag in tags:
                ctype = tag.getAttribute('type')
                data = tag.firstChild.data
                if ctype == 'taxonomic_classifier':
                    #for t in data:
                    if 'Top/News/' in data:
                        tmp = data.strip().split('/')
                        category = '/'.join(tmp[:3])
                        if category in keys:
                            type_cnt[category] += 1
                            #sys.exit(-1)
                        #tags_count[data] += 1
                            tag_names.append(data)
                    #new_set = set()
                    #for i in desc[ctype]:
                    #    if i not in data:
                    #        new_set.add(i)
                    #desc[ctype] = new_set
                #desc[ctype].add(data)
            #print(len(desc))
            #print(desc)
            #print()
            if len(tag_names) == 0:
                continue
            _corpus['type'] = tag_names
            '''
            if len(tag_names) > 0:
                desired_key = sorted(tag_names, key=lambda x:len(x))
                _corpus['type'] = []
                _corpus['type'].append(desired_key[0])
                if len(desired_key) > 1:
                    _corpus['type'].append(desired_key[1])
                ct += 1
            else:
                continue
            '''
            '''
            print('*' * 20)
            ct += 1
            for tag in tags:
                data = tag.childNodes[1].childNodes[0].data
                print(f'abstract: {data}')
            '''
            tags = root.getElementsByTagName('title')
            for tag in tags:
                data = tag.firstChild.data
                _corpus['title'] = data
                #print(f'title: {data}')
            _corpus['id'] = _id
            tags = root.getElementsByTagName('block')
            for tag in tags:
                bclass = tag.getAttribute('class')
                # remove lead_paragraph
                if bclass == 'full_text':
                    ps = tag.getElementsByTagName('p')
                    for p in ps:
                        data = p.firstChild.data
                        _corpus['content'].append(data)
                elif bclass == 'lead_paragraph':
                    ps = tag.getElementsByTagName('p')
                    for p in ps:
                        data = p.firstChild.data
                        _corpus['lead_3'].append(data)
                        #print(data)
            OUT_JSON.write(json.dumps(_corpus) + '\n')
            #sys.exit(-1)
         
'''
    with open('type_stats_business.txt', 'w') as OUT:
        for _type in tags_count:
            OUT.write("{}\t{}\n".format(_type, tags_count[_type]))
'''
print(type_cnt)