import os
import ast
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
import random
import collections
from scipy.stats import pearsonr
random.seed(9001)

import csv

ROOT_DIR = os.getenv('ROOT_DIR', 'default')

encoders = ['ance', 'contriever', 'dpr', 'gtr', 'simcse', 'tasb']

def percent_round(_value):
    return round(_value * 100, 1)

def append_to_result(result, file):
    if os.path.exists(file):
        append_write = 'a'
    else:
        append_write = 'w'

    if isinstance(result, dict):
        result = json.dumps(result)

    with open(file, append_write) as w:
        w.write(result + '\n')


def evaluate_qrels(qid_pids_dict, qrels_dict, topk=5):
    """
    Evaluate the retrieval results based on the qrels
    qid_pids_dict: 
    """
    recalls = []
    for _qid, _pids in qid_pids_dict.items():
        _pids = remove_duplicates_preserve_order(_pids)
        gold_pid = qrels_dict[_qid]
        recalls.append(int(gold_pid in _pids[:topk]))
        
    return round(np.mean(recalls) * 100, 2)


def load_qrels(qrels_file):
    reader = csv.reader(open(qrels_file, encoding="utf-8"),
                        delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    
    next(reader)
    
    qrels = {}
    
    for _id, _row in enumerate(reader):
        query_id, corpus_id, score = _row[0], _row[1], int(_row[2])
        
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
            
    return qrels

def get_records(file_path, searcher=None):
    """
        Get the records from the file
        searcher is a pyserini searcher
        There will be two types of inputs:
        1. .aug: qid, pid, content
        2. .txt: the pyserini retrieval result

        Return: List of Dict
        [{
            'qid': str,
            'pid': str,
            'content': str
        }]
    """

    if file_path.endswith('.aug') or file_path.endswith('tsv') or 'rrf' in file_path:
        #read pd dataframe
        records = pd.read_csv(file_path, sep='\t').to_dict('records')
        if searcher == None:
            return records

        if 'content' not in records[0]:
            for record in records:
                doc = searcher.doc(record['pid'])
                content = json.loads(doc.raw())['contents'].replace('\n', '\\n').replace('\t', ' ')
                record['content'] = content

    elif file_path.endswith('.txt'):
        records = []
        list_records = pd.read_csv(file_path, sep=' ', header=None).to_records()
        for record in tqdm(list_records):
            if searcher is not None:
                doc = searcher.doc(record[3])
                content = json.loads(doc.raw())['contents'].replace('\n', '\\n').replace('\t', ' ')
            else:
                content = None
            record = {
                'qid': record[1],
                'pid': record[3],
                'score': record[5],
                'content': content,
            }
            records.append(record)
    
    return records

def preprocess_beir_dataset(retrieval_dir):
    # process the corpus
    corpus_file = os.path.join(retrieval_dir, 'corpus.jsonl')
    reformat_lines = []
    with open(corpus_file, 'r') as f:
        corpus = [json.loads(line) for line in f.readlines()]
        for line in corpus:
            reformat_lines.append({
                'id': line['_id'],
                'contents': line['title'] + '\n' + line['text'].replace('\n', '\\n')
            })

    write_lines = '\n'.join([json.dumps(line) for line in reformat_lines])
    write_file = os.path.join(retrieval_dir, 'corpus.reformat.jsonl')
    append_to_result(write_lines, write_file)
    # then, we can pass it to the chunk.py for corpus preprocessing
    
    # process the query
    query_file = os.path.join(retrieval_dir, 'queries.jsonl')
    reformat_lines = []
    with open(query_file, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
        for line in lines:
            reformat_lines.append({
                'id': line['_id'],
                'title': line['text'],
            })

    write_lines = '\n'.join([json.dumps(line) for line in reformat_lines])
    write_file = os.path.join(retrieval_dir, "queries.reformat.jsonl")
    append_to_result(write_lines, write_file)


def process_decomposed_query(query_decomposition_file):
    retrieval_dir = os.path.dirname(query_decomposition_file)
    whole_file = os.path.join(retrieval_dir, 'queries.whole.jsonl')
    multi_file = os.path.join(retrieval_dir, 'queries.multi.jsonl')

    whole_lines = []
    multi_lines = []
    
    with open(query_decomposition_file, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
        for line in lines:
            if len(line['prop_generation']) <= 1:
                continue
            
            _id = line['id']
            _title = line['input']
            
            whole_lines.append({
                'id': _id,
                'title': _title,
            })
            
            for _idx, prop in enumerate(line['prop_generation']):
                multi_lines.append({
                    'id': _id + '#' + str(_idx),
                    'title': prop,
                })
                
    write_lines = '\n'.join([json.dumps(line) for line in whole_lines])
    append_to_result(write_lines, whole_file)

    write_lines = '\n'.join([json.dumps(line) for line in multi_lines])
    append_to_result(write_lines, multi_file)