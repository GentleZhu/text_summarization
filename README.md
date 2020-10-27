# SUMDocS: Surrounding-aware Unsupervised Multi-Document Summarization
## What is surrounding-aware document summarization?
Existing multi-document summarization methods ignore the fact that there often exist many relevant documents that provide surrounding (topical related) background knowledge, which can help generate a salient and discriminative summary for a given set of documents.


## Run
To run sumdocs, you need to write a config files like test_config_news and test_config_NLP. The most important parameters are $input_file, $background_file, $output_file and $summ_method.
```
python sumdocs.py $config_file
```

Test the Multi-News dataset
```
python sumdocs.py test_config_news
```

Test the Scientific-NLP dataset
```
python sumdocs.py test_config_NLP
```

## Baseline and ablations
Set summ_method in config files 

## Evaluation
Multi-facet summarization annotation protocol, details TBD.
