from scipy import spatial
import math
from IPython import embed

from utils import *

def calculate_repr_score(i, sentence_vectors, delta):
    sent_num = len(sentence_vectors)
    current_v = sentence_vectors[i]
    score = 0
    for j in range(sent_num):
        if j == i:
            continue
        v = sentence_vectors[j]
        sim = 1 - spatial.distance.cosine(current_v, v)
        score += int(sim > delta)
    score /= sent_num
    return score

def calculate_div_score(i, sentence_vectors, repr_scores):
    higher_index = set()
    sent_num = len(sentence_vectors)
    for j in range(sent_num):
        if j == i:
            continue
        if repr_scores[j] > repr_scores[i]:
            higher_index.add(j)
    if len(higher_index) > 0:
        sim_scores = [1 - spatial.distance.cosine(sentence_vectors[j], sentence_vectors[i]) for j in higher_index]
        score = 1 - max(sim_scores)
    else:
        sim_scores = [1 - spatial.distance.cosine(sentence_vectors[j], sentence_vectors[i]) \
                      for j in range(sent_num) if j != i]
        score = 1 - min(sim_scores)
    return score

def calculate_len_score(i, el, rl):
    score = el[i] / max(el) * math.log(max(rl) / rl[i])
    return score

def select_sentences(budget, scores, sentences):
    sent_num = len(sentences)
    chosen = []
    current_len = 0
    ranked_list = [(i, scores[i]) for i in scores]
    ranked_list = sorted(ranked_list, key=lambda t: -t[1])
    for i in range(sent_num):
        index = ranked_list[i][0]
        if len(sentences[index]) + current_len > budget:
            continue
        current_len += len(sentences[index])
        chosen.append(index)
    return chosen

def main():
    delta = 0.25
    budget = 665
    stopword_path = '/shared/data/qiz3/text_summ/data/stopwords.txt'
    #corpusIn = open('/shared/data/qiz3/text_summ/data/NYT_annotated_corpus/NYT_corpus.txt')
    #target_set = [5804, 5803, 17361, 20859, 18942, 18336, 21233, 19615, 17945]
    #passages, raw_sentences = load_corpus(corpusIn, target_set, stopword_path)
    for s in duc_set:
        if os.path.exists('res/' + s + '.txt'):
            continue
        passages, raw_sentences = generate_duc_docs(s, stopword_path)

        sent_num = len(passages)
        el, rl = calculate_len(passages, raw_sentences)
        sentence_vecs = construct_sentence_vec(passages)
        repr_scores = {}
        len_scores = {}
        div_scores = {}
        scores = {}
        for i in tqdm(range(sent_num)):
            repr_scores[i] = calculate_repr_score(i, sentence_vecs, delta)
            len_scores[i] = calculate_len_score(i, el, rl)
        for i in tqdm(range(sent_num)):
            div_scores[i] = calculate_div_score(i, sentence_vecs, repr_scores)
            #scores[i] = repr_scores[i]
            scores[i] = math.log(div_scores[i] + 0.01) + math.log(len_scores[i] + 0.01) + math.log(repr_scores[i] + 0.01)

        chosen = select_sentences(budget, scores, raw_sentences)
        summary = ''
        for i in chosen:
            summary += ' ' + raw_sentences[i]
        f = open('res/' + s + '.txt', 'w')
        f.write(summary)
        f.close()

if __name__ == '__main__':
    main()