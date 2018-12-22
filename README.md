# text_summarization
## What is a concept
Concept is hierarchical relationships between entities, which is defined as <root_node, list of relations(top-down)>, e.g. <sports team, [subclas of, instance of]>.
One topdown path of above concept could be `sports team - Basketball Team - Los Angels Lakers`
## How to use concept to psedo-link a corpus
```
concept = Concept(h)
concept.construct_concepts([(u'type of sport', 2), [2,4]])
```
