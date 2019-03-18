import numpy as np
import math
from IPython import embed
from tqdm import tqdm

class phraseExtractor:
    def __init__(self, entity_candidates, phrase2idx, target_doc_list, sibling_groups, entity2freq):
        self.entity_candidates = entity_candidates
        self.target_docs = target_doc_list
        self.sibling_groups = sibling_groups
        self.phrase2idx = phrase2idx
        self.entity2embed = None#entity2embed
        self.entity2freq = entity2freq

        self.ranked_list = []

    def read_int(self, file_path):
        self.int_score = {}
        with open(file_path) as IN:
            for line in IN:
                score, phrase = line.strip('\n').split('\t')
                score = float(score)
                phrase = phrase.replace(' ', '_')
                self.int_score[phrase] = score

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

    def _get_sibling_phrase_cnt(self, target_doc_ids):
        s = 0
        for doc_id in target_doc_ids:
            s += sum([self.document_phrase_cnt[doc_id][entity] for entity in self.document_phrase_cnt[doc_id]])
        return s

    def _get_target_phrase_cnt(self, target_docs):
        s = 0
        for doc in target_docs:
            s += len(doc)
        return s

    def _calculate_max_df(self, sibling_group):
        #print('Collecting group entities...')
        entities = set()
        for doc_id in sibling_group:
            entities |= set(self.document_phrase_cnt[doc_id].keys())
        max_df = -1
        #print('Done. Calculating max df...')
        for entity in entities:
            df = sum([1 if doc_id in self.inverted_index[entity] else 0 for doc_id in sibling_group])
            if df > max_df:
                max_df = df
        #print('Done.')
        return max_df

    def _calculate_target_max_df(self, target_docs):
        entities = set()
        for doc in target_docs:
            entities |= set(doc)
        max_df = -1
        #print('Done. Calculating max df...')
        for entity in entities:
            df = sum([1 if entity in x else 0 for x in target_docs])
            if df > max_df:
                max_df = df
        #print('Done.')
        assert max_df > 0
        return max_df 

    def load_freq_data(self, document_phrase_cnt, inverted_index, option='id'):
        #option->raw: list of strings, id: list of ids
        self.document_phrase_cnt = document_phrase_cnt
        self.inverted_index = inverted_index

        #TODO(@jingjing): below lines need a swtich, one for label-expansion, one for summarization purpose
        #self.avg_dl = sum([self._get_sibling_phrase_cnt(sibling_cell) for sibling_cell in self.sibling_groups])
        if option == 'id':
            self.avg_dl = sum([self._get_sibling_phrase_cnt(sibling_cell) for sibling_cell in self.sibling_groups])
            self.avg_dl += self._get_sibling_phrase_cnt(self.target_docs)
            self.avg_dl /= (len(self.sibling_groups) + 1)
        elif option == 'raw':
            self.avg_dl = self._get_target_phrase_cnt(self.target_docs)
            self.avg_dl += self._get_sibling_phrase_cnt(self.target_docs)
            self.avg_dl /= (len(self.sibling_groups) + 1)
        else:
            raise Exception

    def _calculate_sibling_max_df(self, option='id'):
        # option->raw: list of strings, id: list of ids
        #TODO(@jingjing): below lines need a swtich, see above
        if option == 'id':
            max_dfs = [self._calculate_max_df(self.target_docs)]
            for sibling in self.sibling_groups:
                max_dfs.append(self._calculate_max_df(sibling))
            self.max_dfs = max_dfs
        elif option == 'raw':
            max_dfs = [self._calculate_target_max_df(self.target_docs)]
            #max_dfs = [self._calculate_max_df(self.target_docs)]
            for sibling in self.sibling_groups:
                max_dfs.append(self._calculate_max_df(sibling))
            self.max_dfs = max_dfs
        else:
            raise Exception

    def _calculate_tf(self, entity, group):
        tf = sum([self.inverted_index[entity][doc_id] for doc_id in group])
        return tf

    def _calculate_target_tf(self, entity, target_docs):
        tf = sum([target_docs.count(entity) for doc in target_docs])
        return tf

    def _compute_entity_score(self, entity, score_option):
        current_df = sum([1 if entity in doc else 0 for doc in self.target_docs])
        score_list = []
        context_group = [(self._get_target_phrase_cnt(self.target_docs), current_df)]
        for sibling_group in self.sibling_groups:
            df = sum([1 if self.inverted_index[entity][doc_id] > 0 else 0 for doc_id in sibling_group])
            context_group.append((self._get_sibling_phrase_cnt(sibling_group), df))
        for idx, g in enumerate(context_group):
            if idx == 0:
                score_list.append(self.bm25_df_paper(g[1], self.max_dfs[idx],
                                                     self._calculate_target_tf(entity, self.target_docs), g[0], self.avg_dl))
            else:
                score_list.append(self.bm25_df_paper(g[1], self.max_dfs[idx],
                                                     self._calculate_tf(entity, self.sibling_groups[idx - 1]), g[0], self.avg_dl))
        score_list = self.softmax_paper(score_list)

        distinctiveness = score_list[0]
        popularity = math.log(1 + self.entity2freq[entity], 2)
        if score_option == 'A':
            score = popularity * distinctiveness
        elif score_option == 'B':
            score = distinctiveness
        elif score_option == 'C':
            score = popularity
        elif score_option == 'D':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = popularity * distinctiveness * int_score
        elif score_option == 'E':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = int_score
        elif score_option == 'F':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = int_score * distinctiveness
        elif score_option == 'H':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = int_score * popularity
        else:
            raise NotImplementedError
        return score

    # this function is for label expansion
    def __compute_entity_score(self, entity, score_option):
        current_df = sum([1 if self.inverted_index[entity][doc_id] > 0 else 0 for doc_id in self.target_docs])
        score_list = []
        context_group = [(self._get_sibling_phrase_cnt(self.target_docs), current_df)]
        for sibling_group in self.sibling_groups:
            df = sum([1 if self.inverted_index[entity][doc_id] > 0 else 0 for doc_id in sibling_group])
            context_group.append((self._get_sibling_phrase_cnt(sibling_group), df))
        for idx, g in enumerate(context_group):
            if idx == 0:
                score_list.append(self.bm25_df_paper(g[1], self.max_dfs[idx],
                                                     self._calculate_tf(entity, self.target_docs), g[0], self.avg_dl))
            else:
                score_list.append(self.bm25_df_paper(g[1], self.max_dfs[idx],
                                                     self._calculate_tf(entity, self.sibling_groups[idx - 1]), g[0], self.avg_dl))
        score_list = self.softmax_paper(score_list)

        distinctiveness = score_list[0]
        popularity = math.log(1 + self.entity2freq[entity], 2)
        if score_option == 'A':
            score = popularity * distinctiveness
        elif score_option == 'B':
            score = distinctiveness
        elif score_option == 'C':
            score = popularity
        elif score_option == 'D':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = popularity * distinctiveness * int_score
        elif score_option == 'E':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = int_score
        elif score_option == 'F':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = int_score * distinctiveness
        elif score_option == 'H':
            int_score = self.int_score[entity] if entity in self.int_score else 0.3
            score = int_score * popularity
        else:
            raise NotImplementedError
        return score

    def compute_scores(self, document_phrase_cnt, inverted_index, score_option, mode = 0):
        # score_option: A. popularity+distinc;
        #               B. distinc
        #               C. popularity
        #               D. Integrity + popularity + distinc
        #               E. Integrity
        #               F. Integrity + distinc
        #               G. tf-idf
        #               H. Integrity + popularity
        self.load_freq_data(document_phrase_cnt, inverted_index)
        self._calculate_sibling_max_df()
        if score_option in ['D', 'E', 'F', 'H']:
            self.read_int('/shared/data/qiz3/text_summ/src/jt_code/HiExpan-master/data/full/intermediate/AutoPhrase.txt')
        #print('Start calculating scores...')
        for entity in self.entity_candidates:
            if score_option == 'G':
                score = math.log(len(document_phrase_cnt)) / math.log(1 + len(inverted_index[entity])) * self.entity2freq[entity]
                #if '_' not in entity:
                #    score = 0.
            else:
                if mode == 0:
                    score = self._compute_entity_score(entity, score_option)
                elif mode == 1:
                    score = self.__compute_entity_score(entity, score_option)
            self.ranked_list.append((entity, score))

        self.ranked_list = sorted(self.ranked_list, key=lambda t: -t[1])
        return self.ranked_list
