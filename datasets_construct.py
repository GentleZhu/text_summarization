import sys,json

#with open(sys.argv[1]) as IN, open(sys.argv[2], 'w', encoding='utf8') as OUT:
#	for line in IN:
#		OUT.write(' '.join(json.loads(line)['content'][:3]) + '\n')
sub_types = []
with open(sys.argv[1]) as IN:
	results = []
	for idx,line in enumerate(IN):
		tmp = json.loads(line)
		if len(tmp['type']) > 1:
			sub_type = tmp['type'][1].lstrip('Top/News/Sports/').split('/')[0]
			if sub_type not in sub_types:
				sub_types.append(sub_type)
			results.append([idx, sub_types.index(sub_type)])
	print(sub_types, len(results))