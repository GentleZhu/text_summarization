import xml.dom.minidom
from collections import defaultdict
from tqdm import tqdm
import sys, json

ct = 0
with open("/shared/data/yuningm2/datasets/NYT_annotated_corpus/data/filelist_2003-07.txt") as f, open('NYT_business.json', 'w') as OUT_JSON:
    tags_count = defaultdict(int)
    for line in tqdm(f.readlines()):
        xml_path = "/shared/data/yuningm2/datasets/NYT_annotated_corpus/data/accum2003-07/" + line.strip()
        #with open(xml_path) as f2:
            #print(f2.readlines())
        dom = xml.dom.minidom.parse(xml_path)
        root = dom.documentElement
        
        tags = root.getElementsByTagName('abstract')
        if len(tags) > 0:
            
            _corpus = {'abstract':[], 'title':[], 'type':[], 'content':[]}
            for tag in tags:
                try:
                    data = tag.childNodes[1].childNodes[0].data
                    _corpus['abstract'].append(data) 
                except:
                    pass

            tags = root.getElementsByTagName('classifier')
            #desc = defaultdict(set)
            tag_names = []
            for tag in tags:
                ctype = tag.getAttribute('type')
                data = tag.firstChild.data
                if ctype == 'taxonomic_classifier':
                    #for t in data:
                    if 'Business' in data:
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
            if len(tag_names) > 0:
                desired_key = sorted(tag_names, key=lambda x:len(x))
                _corpus['type'] = []
                tags_count[desired_key[0]] += 1
                _corpus['type'].append(desired_key[0])
                if len(desired_key) > 1:
                    tags_count[desired_key[1]] += 1
                    _corpus['type'].append(desired_key[1])
                ct += 1
            else:
                continue
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
                _corpus['title'].append(data)
                #print(f'title: {data}')

            tags = root.getElementsByTagName('block')
            for tag in tags:
                bclass = tag.getAttribute('class')
                # remove lead_paragraph
                if bclass == 'full_text':
                    ps = tag.getElementsByTagName('p')
                    for p in ps:
                        data = p.firstChild.data
                        _corpus['content'].append(data)
                        #print(data)
            OUT_JSON.write(json.dumps(_corpus) + '\n')

            
                
        #if ct == 200:
        #    break
        #    print(tags_count)
        #    exit()
    with open('type_stats_business.txt', 'w') as OUT:
        for _type in tags_count:
            OUT.write("{}\t{}\n".format(_type, tags_count[_type]))