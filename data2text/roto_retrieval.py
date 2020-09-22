from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import os
import json
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from pprint import pprint
from multiprocessing import Pool
import munkres
from sklearn.feature_extraction.text import TfidfVectorizer
from text2num import text2num, NumberException

from tqdm import tqdm

try:
    range = xrange
except NameError:
    pass

from data_utils import get_train_ents, extract_entities, extract_numbers


# ignore_rels = ['HOME_AWAY', 'TEAM_NAME', 'PLAYER_NAME']
ignore_rels = []
LARGE_NUM = 10000000


# load all entities
all_ents, players, teams, cities = get_train_ents()


class Record(object):
    def __init__(self, rel, entity, value):
        self.rel = rel
        self.entity = entity
        self.value = value

    def __str__(self):
        return "{}|{}|{}".format(self.rel, self.entity, self.value)


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class RecordDataset(object):
    def __init__(self, src_file, tgt_file):
        self.records = []
        self.normalized_targets = []
        with open(src_file) as srcf, open(tgt_file) as tgtf:
            idx = 0
            for line_src, line_tgt in zip(srcf, tgtf):
                ls = line_src.strip().split()
                if ls:
                    records = []
                    for triple in ls:
                        value, rel, entity = triple.split('|')
                        try:
                            value = int(value)
                        except ValueError:
                            value = str(value)
                        r = Record(rel, entity, value)
                        records.append(r)
                    self.records.append({'idx': idx, 'records': records, 'target': line_tgt.strip()})
                    idx += 1

        print("Number of records and tgts read: ", len(self.records))
        self.normalize_text()
        self.target_token_set = self.create_token_set()
        self.sklearn_tfidf = TfidfVectorizer(sublinear_tf=True, tokenizer=lambda x: x, preprocessor=lambda x: x,
                                             lowercase=False, token_pattern=None)
        self.tfidf_score = self.sklearn_tfidf.fit_transform(self.normalized_targets).toarray()

    def normalize_text(self):
        for r in self.records:
            tgt_tokens = r['target'].split()
            tgt_tokens = self.normalize_sent(tgt_tokens)
            self.normalized_targets.append(tgt_tokens)

    def create_token_set(self):
        token_sets = []
        for sent in self.normalized_targets:
            token_sets.append(set(sent))
        return token_sets

    @staticmethod
    def normalize_sent(tgt):
        ents = extract_entities(tgt, all_ents)
        nums = extract_numbers(tgt)
        ranges = []
        for ent in ents:
            ranges.append((ent[0], ent[1], 'ENT'))
        for num in nums:
            ranges.append((num[0], num[1], 'NUM'))
        ranges.sort(key=lambda x: x[0])

        masked_sent = []
        i = 0
        while i < len(tgt):
            match = False
            for r in ranges:
                if i == r[0]:
                    match = True
                    masked_sent.append(r[2])
                    i = r[1]
                    break
            if not match:
                masked_sent.append(tgt[i])
                i += 1
        return masked_sent

    @staticmethod
    def generalized_jaccard(a, b):
        a, b = map(Counter, (a, b))
        return sum((a & b).values()) / sum((a | b).values())

    @staticmethod
    def construct_cost_matrix(query, target):
        cost_matrix = [[LARGE_NUM] * len(target) for _ in range(len(query))]
        for i, q in enumerate(query):
            for j, t in enumerate(target):
                if q.rel == t.rel:
                    if isinstance(q.value, str) and isinstance(t.value, str):
                        if q.value == t.value:
                            cost_matrix[i][j] = 0
                        else:
                            cost_matrix[i][j] = 1
                    elif isinstance(q.value, int) and isinstance(t.value, int):
                        cost_matrix[i][j] = abs(q.value - t.value)
        return cost_matrix

    def calculate_score(self, query_record, target_record):
        query_rels, target_rels = (
            [r.rel for r in record['records'] if r.rel not in ignore_rels]
            for record in (query_record, target_record))
        return self.generalized_jaccard(query_rels, target_rels)

    def calculate_alignment(self, a_records, b_records):
        km = munkres.Munkres()
        cost_matrix = self.construct_cost_matrix(a_records, b_records)
        indexes = km.compute(cost_matrix)
        total_cost = 0
        filtered_indexes = []
        for row, column in indexes:
            value = cost_matrix[row][column]
            if value < LARGE_NUM:
                filtered_indexes.append((row, column))
                total_cost += value
        #     print('(%d, %d) -> %d' % (row, column, value))
        # print("total cost: %d" % total_cost)
        return filtered_indexes, total_cost

    def retrieve(self, query_record, filter_complete_matching=False, topk=1):
        scores = []
        for example in self.records:
            if example['target'] != query_record['target']:
                jaccard = self.calculate_score(query_record, example)
                scores.append((jaccard, example))
        if filter_complete_matching:
            scores = list(filter(lambda x: x[0] < 1, scores))
        highest_score = max(map(lambda x: x[0], scores))
        highest_examples = []
        for s in filter(lambda s: s[0] == highest_score, scores):
            # start calculating the min cost
            filtered_indexes, total_cost = self.calculate_alignment(query_record['records'], s[1]['records'])
            highest_examples.append({'jaccard': s[0], 'min_cost': total_cost, 'alignment': filtered_indexes,
                                     'retrieved': s[1], 'query': query_record})
        highest_examples.sort(key=lambda x: x['min_cost'])
        return highest_examples[0]

    def retrieve_with_target(self, query_record):
        query_tgt_tokens = query_record['target'].split()
        query_tgt_tokens = self.normalize_sent(query_tgt_tokens)
        query_token_set = set(query_tgt_tokens)

        scores = []
        for idx, tokens in enumerate(self.target_token_set):
            if query_record['target'] != self.records[idx]['target']:
                # scores.append((idx, sentence_bleu([query_tgt_tokens], tokens)))
                # scores.append((idx, self.generalized_jaccard(tokens, query_tgt_tokens)))
                score = 0
                for token in tokens:
                    if token in query_token_set:
                        score += self.tfidf_score[idx, self.sklearn_tfidf.vocabulary_[token]]
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        filtered_indexes, total_cost = self.calculate_alignment(query_record['records'],
                                                                self.records[scores[0][0]]['records'])
        return {'score': scores[0][1], 'cost': total_cost, 'alignment': filtered_indexes,
                'retrieved': self.records[scores[0][0]], 'query': query_record}

    def retrieve_with_target_bleu(self, query_record):
        query_tgt_tokens = query_record['target'].split()
        query_tgt_tokens = self.normalize_sent(query_tgt_tokens)

        scores = []
        for idx, tokens in enumerate(self.normalized_targets):
            if query_record['target'] != self.records[idx]['target']:
                scores.append((idx, sentence_bleu([query_tgt_tokens], tokens)))
        scores.sort(key=lambda x: x[1], reverse=True)
        filtered_indexes, total_cost = self.calculate_alignment(query_record['records'],
                                                                self.records[scores[0][0]]['records'])
        return {'score': scores[0][1], 'cost': total_cost, 'alignment': filtered_indexes,
                'retrieved': self.records[scores[0][0]], 'query': query_record}


def retrieve(query_record):
    return rd.retrieve(query_record, filter_complete_matching=True)


def retrieve_with_target(query_record):
    return rd.retrieve_with_target(query_record)


def retrieve_with_target_bleu(query_record):
    return rd.retrieve_with_target_bleu(query_record)


def main(rd_prefix, use_target, use_bleu):
    global rd

    stages = ['train', 'valid', 'test']

    rds = {stage: RecordDataset(
        '{}{}.src'.format(rd_prefix, stage),
        '{}{}.tgt'.format(rd_prefix, stage))
        for stage in stages}
    rd = rds['train']

    if not use_target:
        retrieve_fn = retrieve
        retrieved_name = 'retrieved'

    else:
        if not use_bleu:
            retrieve_fn = retrieve_with_target
            retrieved_name = 'retrieved_target'

        else:
            retrieve_fn = retrieve_with_target_bleu
            retrieved_name = 'retrieved_target_bleu'

    for stage in stages:
        print('Start {} retrieval'.format(stage))
        with Pool(24) as p:
            retrieved = p.map(retrieve_fn, rds[stage].records)
        with open('{}_{}.json'.format(retrieved_name, stage), 'w') as of:
            json.dump(retrieved, of, cls=MyEncoder)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('rd_prefix')
    argparser.add_argument('--use_target', action='store_true')
    argparser.add_argument('--use_bleu', action='store_true')
    main(**vars(argparser.parse_args()))
