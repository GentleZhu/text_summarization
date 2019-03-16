# Background-Aware Multi-doc Summarization
## What is a concept
Concept is hierarchical relationships between entities, which is defined as <root_node, list of relations(top-down)>, e.g. <sports team, [subclas of, instance of]>.
In this work, we simplify the relation only hypernym, i.e. *is a*.
One topdown path of above concept could be `Science - Astronomy - planet`
## How to use concept to psedo-link a corpus
```
concept = Concept(h)
concept.construct_concepts([(u'type of sport', 2), [2,4]])
```

## Configuration
On dmserv5, using qiz3
```
source activate kprnn
```

## Iteratively document categorization and label expansion
### Document categorization
parameters: {expan: 0/1/2...}, 0 means link using original hierarchy, 1/2/.. will load. preprocess=True means link the corpus, otherwise use dumped training data to train embedding directly. 
```
python3 summ_pipeline.py train_config
```
### Label expansion
First set expan = 0, it will store new concepts with suffix expan = 1, see code for details.
```
python3 summ_pipeline.py examine_config
```
After expansion, re-run document categorization with expan += 1 in config.

## Comparative Summarization
```
python3 summ_pipeline.py test_config
```
Current default setting 'comparative_opt' = KNN, which is route0 and route1.

## Evaluation
Multi-facet summarization annotation protocol, details TBD.
