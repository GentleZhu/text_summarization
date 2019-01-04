# text_summarization
## What is a concept
Concept is hierarchical relationships between entities, which is defined as <root_node, list of relations(top-down)>, e.g. <sports team, [subclas of, instance of]>.
One topdown path of above concept could be `sports team - Basketball Team - Los Angels Lakers`
## How to use concept to psedo-link a corpus
```
concept = Concept(h)
concept.construct_concepts([(u'type of sport', 2), [2,4]])
```

## Run and configuration
```
python summ_pipeline.py
```

## Assign background corpus to category of interest
If we want to assign documents to one dimension, for example, sports "baseball, basketball, football and etc.". We will just need top@k relevant documents to these category nodes.

## Conduct comparative analysis between twin and target sets.
## Currently the output is just ranked list based on w_i and n_i, and haven't implemented redundancy part.
```
python summarizer.py
```
