"""
Generate the union of the retieved results between query and sub-query.
query = [p1, p2, ...] 

"""

import os
import ast
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
import random
import collections
random.seed(9001)

import argparse

from helpers import init_query_encoder, init_document_encoder


from pyserini.search.lucene import LuceneSearcher

query_encoder_dict = {
    'ance': 'castorini/ance-dpr-question-multi',
    'contriever': 'facebook/contriever',
    'dpr': 'facebook/dpr-question_encoder-multiset-base',
    'gtr': 'sentence-transformers/gtr-t5-base',
    'simcse': 'princeton-nlp/unsup-simcse-bert-base-uncased',
    'tasb': 'sentence-transformers/msmarco-distilbert-base-tas-b',
}

doc_encoder_dict = {
    'ance': 'castorini/ance-dpr-context-multi',
    'contriever': 'facebook/contriever',
    'dpr': 'facebook/dpr-ctx_encoder-multiset-base',
    'gtr': 'sentence-transformers/gtr-t5-base',
    'simcse': 'princeton-nlp/unsup-simcse-bert-base-uncased',
    'tasb': 'sentence-transformers/msmarco-distilbert-base-tas-b',
}

def get_passage_id_func(args):
    if 'factoid' in args.subquery_file or 'fever' in args.subquery_file:
        print('Using Factoid')
        return lambda x: x[:-5]
    else:
        return lambda x: '-'.join(x.split('-')[:-1])
    
def filter_by_theshold(records, threshold):
    """
    Get the threshold for the records
    """
    qid_list_dict = defaultdict(list)
    for line in records:
        qid = line['qid']
        qid_list_dict[qid].append(line)

    for qid, lines in qid_list_dict.items():
        qid_list_dict[qid] = sorted(lines, key=lambda x: x['score'], reverse=True)[:threshold]

    records = []
    for qid, lines in qid_list_dict.items():
        records.extend(lines)

    return records


def get_records(file_path, searcher=None, threshold=-1):
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

    if os.path.exists(file_path) is False:
        return []

    if file_path.endswith('.aug'):
        #read pd dataframe
        records = pd.read_csv(file_path, sep='\t').to_dict('records')
    elif file_path.endswith('.txt'):
        # for the pyserini retrieval output
        records = []
        list_records = pd.read_csv(file_path, sep=' ', header=None).to_records()
        for record in tqdm(list_records):
            doc = searcher.doc(record[3])
            content = json.loads(doc.raw())['contents'].replace('\n', '\\n').replace('\t', ' ')
            record = {
                'qid': record[1],
                'pid': record[3],
                'score': record[5],
                'content': content,
            }
            records.append(record)

    if threshold > 0:
        records = filter_by_theshold(records, threshold)
    
    return records

def main(args):
    save_dir = os.path.dirname(args.whole_prop_path)
    passage_id_func = get_passage_id_func(args)
    
    chunk_searcher = LuceneSearcher(args.chunk_bm25_dir)
    prop_searcher = LuceneSearcher(args.prop_bm25_dir)

    IS_CONDQA = 'condqa' in args.subquery_file
    # we need a mapping from query to sub-query
    subquery_file = args.subquery_file
    qid_subqid_dict = defaultdict(list)
    subqid_content_dict = {}
    with open(subquery_file, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            subqid = line['id']
            if IS_CONDQA:
                qid = subqid.split('%')[0]
            else:
                qid = subqid.split('#')[0]
            qid_subqid_dict[qid].append(subqid)
            subqid_content_dict[subqid] = line['title']

    qid_content_dict = {}
    query_file = args.query_file
    with open(query_file, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            qid = line['id']
            if qid not in qid_subqid_dict:
                continue
            
            query = line['title']
            qid_content_dict[qid] = query

    qid_content_dict = {**qid_content_dict, **subqid_content_dict}

    # get the dict of passage_prop_dict
    with open(args.passage2prop, 'rb') as f:
        passage2prop = pickle.load(f)
    
    # get the target for retreival
    whole_prop_path = args.whole_prop_path
    multi_prop_path = args.multi_prop_path
    whole_chunk_path = args.whole_chunk_path
    multi_chunk_path = args.multi_chunk_path

    # qid_props_id
    qid_pid_dict = defaultdict(list)
    subqid_pid_dict = defaultdict(list)
    qid_cid_dict = defaultdict(list)
    subqid_cid_dict = defaultdict(list)

    # cid/pid -> text
    pid_content_dict = {}
    qid_pid_pair_existing = []

    whole_prop_ret_records = get_records(whole_prop_path, prop_searcher, args.threshold)    
    for record in whole_prop_ret_records:
        qid = str(record['qid'])
        if qid not in qid_content_dict:
            continue
        pid = record['pid']
        content = record['content']
        pid_content_dict[pid] = content
        qid_pid_dict[qid].append(pid)
        qid_pid_pair_existing.append((qid, pid))

    multi_prop_ret_records = get_records(multi_prop_path, prop_searcher, args.threshold)
    for record in multi_prop_ret_records:
        subqid = str(record['qid'])
        if subqid not in qid_content_dict:
            continue
        pid = record['pid']
        content = record['content']
        pid_content_dict[pid] = content
        subqid_pid_dict[subqid].append(pid)
        qid_pid_pair_existing.append((subqid, pid))

    whole_chunk_ret_records = get_records(whole_chunk_path, chunk_searcher, args.threshold)
    for record in whole_chunk_ret_records:
        qid = str(record['qid'])
        if qid not in qid_content_dict:
            continue
        cid = record['pid']
        content = record['content']
        pid_content_dict[cid] = content
        qid_cid_dict[qid].append(cid)
        qid_pid_pair_existing.append((qid, cid))

    multi_chunk_ret_records = get_records(multi_chunk_path, chunk_searcher, args.threshold)
    for record in multi_chunk_ret_records:
        subqid = str(record['qid'])
        if subqid not in qid_content_dict:
            continue
        cid = record['pid']
        content = record['content']
        pid_content_dict[cid] = content
        subqid_cid_dict[subqid].append(cid)
        qid_pid_pair_existing.append((subqid, cid))

    # encoding pairs
    pid_to_encode = list()

    # need to get all (qid, cid) pair, and then remove the existing ones
    qid_cid_pair = []
    for (_k, _v) in qid_cid_dict.items():
        for _cid in _v:
            qid_cid_pair.append((_k, _cid))
    
    for(_k, _v) in subqid_cid_dict.items():
        for _cid in _v:
            if IS_CONDQA:
                qid_cid_pair.append((_k.split('%')[0], _cid))
            else:
                qid_cid_pair.append((_k.split('#')[0], _cid))

    qid_cid_pair = list(set(qid_cid_pair))

    for (_k, _v) in qid_pid_dict.items():
        for _pid in _v:
            qid_cid_pair.append((_k, passage_id_func(_pid)))

    qid_cid_pair = list(set(qid_cid_pair))

    for (_k, _v) in subqid_pid_dict.items():
        for _pid in _v:
            if IS_CONDQA:
                qid_cid_pair.append((_k.split('%')[0], passage_id_func(_pid)))
            else:
                qid_cid_pair.append((_k.split('#')[0], passage_id_func(_pid)))

    qid_cid_pair = list(set(qid_cid_pair))


    # then we should collect the pair of query/sub-query and passage/propositions
    # add (query, chunk)
    qid_pid_pair_encoding = [(_k, _v) for _k, _v in qid_cid_pair]
    for _k, _v in qid_cid_pair:
        # add (subquery, chunk)
        for subquery_id in qid_subqid_dict[_k]:
            qid_pid_pair_encoding.append((subquery_id, _v))

        # add (query, proposition)
        for prop_id in passage2prop[_v]:
            qid_pid_pair_encoding.append((_k, prop_id))
            # add (subquery, proposition)
            for subquery_id in qid_subqid_dict[_k]:
                qid_pid_pair_encoding.append((subquery_id, prop_id))

    # get the existing pairs
    qid_pid_pair_encoding = list(set(qid_pid_pair_encoding) - set(qid_pid_pair_existing))

    device = args.device
    # initialize encoder for both query and documents
    query_encoder = init_query_encoder(args.query_encoder, None, device)
    doc_encoder = init_document_encoder(args.doc_encoder, None, device)
    query_encoder.model.eval()
    doc_encoder.model.eval()

    # generate the list for queries and docs
    qid_to_encode = list(set(_p[0] for _p in qid_pid_pair_encoding))
    query_to_encode = [qid_content_dict[qid] for qid in qid_to_encode]
    pid_to_encode = list(set(_p[1] for _p in qid_pid_pair_encoding))

    # add the content
    for _pid in tqdm(pid_to_encode, desc='Search content for pid'):
        doc = prop_searcher.doc(_pid)
        if doc is None:
            doc = chunk_searcher.doc(_pid)
        
        try:
            content = json.loads(doc.raw())['contents'].replace('\n', '\\n').replace('\t', ' ')
        except:
            print(_pid)
            raise Exception
        pid_content_dict[_pid] = content
    
    doc_to_encode = [pid_content_dict[pid] for pid in pid_to_encode]
    title_to_encode = [doc.split('\\n')[0] for doc in doc_to_encode]
    text_to_encode = [' '.join(doc.split('\\n')[1:]) for doc in doc_to_encode]

    print('Len query: ', len(qid_to_encode))
    print('Len doc: ', len(doc_to_encode))

    # manually batch operation
    queries_np = np.zeros((len(query_to_encode), 768), dtype=np.float16)
    docs_np = np.zeros((len(title_to_encode), 768), dtype=np.float16)

    # query_batch_num = len(query_to_encode) // args.batch_size + int(len(query_to_encode) % args.batch_size > 0)
    doc_batch_num = len(title_to_encode) // args.batch_size + int(len(title_to_encode) % args.batch_size > 0)

    # for query_batch_index in tqdm(range(query_batch_num)):
    #     query_batch = query_to_encode[query_batch_index * args.batch_size : (query_batch_index + 1) * args.batch_size]
    #     queries_np[query_batch_index*args.batch_size : (query_batch_index + 1)*args.batch_size, :] = query_encoder.encode(query_batch)

    for query_index in tqdm(range(len(query_to_encode))):
        queries_np[query_index] = query_encoder.encode(query_to_encode[query_index])


    for doc_batch_index in tqdm(range(doc_batch_num)):
        text_batch = text_to_encode[doc_batch_index * args.batch_size : (doc_batch_index + 1) * args.batch_size]
        title_batch = title_to_encode[doc_batch_index * args.batch_size : (doc_batch_index + 1) * args.batch_size]

        kwargs = {
            'texts': text_batch,
            'titles': title_batch,
            'expands': None,
            'fp16': True,
            'max_length': 256,
            'add_sep': False,
        }

        docs_np[doc_batch_index*args.batch_size : (doc_batch_index + 1)*args.batch_size, :] = doc_encoder.encode(**kwargs)
    

    qid_to_index = {_q: _i for _i, _q in enumerate(qid_to_encode)}
    pid_to_index = {_p: _i for _i, _p in enumerate(pid_to_encode)}

    lines = []
    qid_pid_list_dict = defaultdict(list)
    for _qid, _pid in qid_pid_pair_encoding:
        qid_pid_list_dict[_qid].append(_pid)

    for (_qid, _pids) in tqdm(qid_pid_list_dict.items(), desc='Calculate scores', total=len(qid_pid_list_dict)):
        qid_index = qid_to_index[_qid]
        pid_indexes = [pid_to_index[_pid] for _pid in _pids]
        scores = np.matmul(docs_np[pid_indexes, :], queries_np[qid_index, :]).tolist()
        
        for (_pid, score) in zip(_pids, scores):
            line = {
                'qid': _qid,
                'pid': _pid,
                'score': score,
            }
            lines.append(line)
    
    save_file = args.multi_prop_path.replace('run.', 'run.quadrant.')
    if save_file.endswith('txt'):
        save_file = save_file + '.aug'
    df = pd.DataFrame(lines)
    df.to_csv(save_file, sep='\t')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder model name or path', required=True)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--prop_bm25_dir', type=str, required=True)
    parser.add_argument('--chunk_bm25_dir', type=str, required=True)
    parser.add_argument('--subquery_file', type=str, required=True, help='subquery jsonl')
    parser.add_argument('--query_file', type=str, required=True, help='query jsonl')
    parser.add_argument('--passage2prop', type=str, required=True, help='mapping from passage to prop')
    parser.add_argument('--whole_prop_path', type=str, required=True)
    parser.add_argument('--multi_prop_path', type=str, required=True)
    parser.add_argument('--whole_chunk_path', type=str, required=True)
    parser.add_argument('--multi_chunk_path', type=str, required=True)
    
    args = parser.parse_args()

    args.query_encoder = query_encoder_dict[args.encoder]
    args.doc_encoder = doc_encoder_dict[args.encoder]

    main(args)