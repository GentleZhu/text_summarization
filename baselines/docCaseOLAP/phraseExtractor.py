import numpy as np
import math
from IPython import embed
from tqdm import tqdm

alpha = 0.5

class phraseExtractor:
    def __init__(self, entity_candidates, target_doc_list, sibling_groups, entity2freq):
        self.entity_candidates = entity_candidates
        self.target_doc_list = target_doc_list
        self.sibling_groups = sibling_groups
        self.entity2embed = None#entity2embed
        self.entity2freq = entity2freq

        self.ranked_list = []

    def bm25_df_paper(self, df, max_df, tf, dl, avgdl, k=1.2, b=0.5, multiplier=3):
        if df * tf == 0:
            return 0
        score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avgdl)))
        df_factor = math.log(1 + df, 2) / math.log(1 + max_df, 2)
        score *= df_factor
        score *= multiplier
        return score

    def softmax_paper(self, score_list):
        # normalization of exp
        exp_sum = 1
        for score in score_list:
            exp_sum += math.exp(score)

        exp_list = []
        for score in score_list:
            normal_value = math.exp(score) / exp_sum
            exp_list.append(normal_value)
        return exp_list

    def _get_target_phrase_cnt(self, target_doc_ids):
        s = 0
        for doc_id in target_doc_ids:
            s += sum([self.document_phrase_cnt['a_' + str(doc_id)][entity] for entity in self.document_phrase_cnt['a_' + str(doc_id)]])
        return s

    def _calculate_max_df(self, sibling_group):
        print('Collecting group entities...')
        entities = set()
        for doc_id in tqdm(sibling_group):
            entities |= self.document_phrase_cnt['a_' + str(doc_id)].keys()
        max_df = -1
        print('Done. Calculating max df...')
        for entity in tqdm(entities):
            df = sum([1 if self.inverted_index[entity]['a_' + str(doc_id)] > 0 else 0 for doc_id in sibling_group])
            if df > max_df:
                max_df = df
        print('Done.')
        return max_df

    def load_freq_data(self, document_phrase_cnt, inverted_index):
        self.document_phrase_cnt = document_phrase_cnt
        self.inverted_index = inverted_index

        self.avg_dl = sum([self._get_target_phrase_cnt(sibling_cell) for sibling_cell in self.sibling_groups])
        self.avg_dl += self._get_target_phrase_cnt(self.target_doc_list)
        self.avg_dl /= (len(self.sibling_groups) + 1)

    def _calculate_sibling_max_df(self):
        max_dfs = [self._calculate_max_df(self.target_doc_list)]
        for sibling in self.sibling_groups:
            max_dfs.append(self._calculate_max_df(sibling))
        self.max_dfs = max_dfs

    def _calculate_tf(self, entity, group):
        tf = sum([self.inverted_index[entity]['a_' + str(doc_id)] for doc_id in group])
        return tf

    def _compute_entity_score(self, entity, with_popularity):
        current_df = sum([1 if self.inverted_index[entity]['a_' + str(doc_id)] > 0 else 0 for doc_id in self.target_doc_list])
        score_list = []
        context_group = [(self._get_target_phrase_cnt(self.target_doc_list), current_df)]
        for sibling_group in self.sibling_groups:
            df = sum([1 if self.inverted_index[entity]['a_' + str(doc_id)] > 0 else 0 for doc_id in sibling_group])
            context_group.append((self._get_target_phrase_cnt(sibling_group), df))
        for idx, g in enumerate(context_group):
            if idx == 0:
                score_list.append(self.bm25_df_paper(g[1], self.max_dfs[idx],
                                                     self._calculate_tf(entity, self.target_doc_list), g[0], self.avg_dl))
            else:
                score_list.append(self.bm25_df_paper(g[1], self.max_dfs[idx],
                                                     self._calculate_tf(entity, self.sibling_groups[idx - 1]), g[0], self.avg_dl))
        score_list = self.softmax_paper(score_list)

        distinctiveness = score_list[0]
        popularity = math.log(1 + self.entity2freq[entity], 2)
        if with_popularity:
            score = popularity * distinctiveness
        else:
            score = distinctiveness
        return score

    def compute_scores(self, document_phrase_cnt, inverted_index, with_popularity=True):
        self.load_freq_data(document_phrase_cnt, inverted_index)
        self._calculate_sibling_max_df()
        print('Start calculating scores...')
        for entity in tqdm(self.entity_candidates):
           score = self._compute_entity_score(entity, with_popularity)
           self.ranked_list.append((entity, score))

        self.ranked_list = sorted(self.ranked_list, key=lambda t: -t[1])
        return self.ranked_list

    def select_phrases(self, target_number):
        '''
        Select phrases one by one, considering redundancy and other scores.
        '''
        phrase_vectors = {}
        for (phrase, _) in self.ranked_list:
            phrase_vectors[phrase] = self.entity2embed[phrase]
        chosen_phrases = []
        phrase_number = len(self.ranked_list)
        chosen = [False for _ in self.ranked_list]
        for _ in tqdm(range(min(target_number, phrase_number))):
            max_score = -9999999
            pick = -1
            for idx, (phrase, score) in enumerate(self.ranked_list):
                tmp_score = score
                for (chosen_phrase, chosen_score) in chosen_phrases:
                    phrase_vector1, phrase_vector2 = phrase_vectors[phrase], phrase_vectors[chosen_phrase]
                    tmp_score_ = score - alpha * np.dot(phrase_vector1, phrase_vector2) / np.linalg.norm(phrase_vector1) \
                                         / np.linalg.norm(phrase_vector2) * chosen_score
                    if tmp_score_ < tmp_score:
                        tmp_score = tmp_score_
                if tmp_score > max_score and not chosen[idx]:
                    max_score = tmp_score
                    pick = idx

            if pick == -1:
                break
            chosen[pick] = True
            chosen_phrases.append(phrase_scores[pick])

        return chosen_phrases