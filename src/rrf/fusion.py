"""
This script is used to calculate the rrf score for the combination in the quadrant.

The quadrants include:
1. whole query and chunks
2. subquery and chunks
3. whole query and propositions
4. subquery and propositions

"""


import os
import ast
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
import random
import collections
random.seed(9001)
import argparse

ROOT_DIR = os.getenv('ROOT_DIR', default)

def append_to_result(result, file):
    if os.path.exists(file):
        append_write = 'a'
    else:
        append_write = 'w'

    if isinstance(result, dict):
        result = json.dumps(result)

    with open(file, append_write) as w:
        w.write(result + '\n')


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

    if file_path.endswith('.aug'):
        #read pd dataframe
        records = pd.read_csv(file_path, sep='\t').to_dict('records')
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
    
    if threshold > 0:
        records = filter_by_theshold(records, threshold)
    
    return records

def answer_passage_hit(answer_list, passage_list):
    """
    answer_list: list of golden answers
    passage_list: list of the contents retrieve
    """
    passage_con = ' '.join(passage_list)
    for answer in answer_list:
        if answer.lower() in passage_con.lower():
            return True
    return False

def calculate_scores(qid_returns, qidpid_score_dict):
    for return_idx in range(len(qid_returns)):
        qid = qid_returns[return_idx]['qid']
        pids = qid_returns[return_idx]['pids']
        subqids = qid_subqid_dict[qid]
        pids_list = [passage2prop[pid] for pid in pids]

        qid_returns[return_idx]['qid_prop_score'] = [
            [qidpid_score_dict[(qid, _prop_id)] for _prop_id in _prop_ids] for _prop_ids in pids_list
        ]

        qid_returns[return_idx]['subqid_prop_score'] = [
            [[qidpid_score_dict[(_subqid, _prop_id)] for _prop_id in _prop_ids] for _subqid in subqids] for _prop_ids in pids_list
        ]

    return qid_returns


def save_results(qid_pid_list_dict, save_file):
    lines = []
    for _qid, _pid_scores in qid_pid_list_dict.items():
        tmp_lines = []
        for _pid, _score in _pid_scores:
            tmp_lines.append({
                'qid': _qid,
                'pid': _pid,
                'score': _score,
            })
        tmp_lines = sorted(tmp_lines, key=lambda x: x['score'], reverse=True)
        lines.extend(tmp_lines)
    
    df = pd.DataFrame(lines)
    df.to_csv(save_file, sep='\t')

def conv_by_encoder(args,
                    whole_chunk_records,
                    whole_prop_records,
                    multi_chunk_records,
                    multi_prop_records,
                    add_query_records,
                    rerank=False,
                    threshold=1000): 
    # record storage
    qid_cid_dict = defaultdict(list)
    subqid_cid_dict = defaultdict(list)
    qid_pid_dict = defaultdict(list)
    subqid_pid_dict = defaultdict(list)
    
    # general pairs for qid and pid
    qidpid_score_dict = dict()
    subqid_set = set([_subqid for _v in qid_subqid_dict.values() for _subqid in _v])
    
    for record in whole_chunk_records:
        qid = str(record['qid'])
        if qid not in qid_subqid_dict:
            continue
        cid = record['pid']
        score = record['score']
        qid_cid_dict[qid].append((cid, score))
        qidpid_score_dict[(qid, cid)] = score

    for record in multi_chunk_records:
        subqid = str(record['qid'])
        if subqid not in subqid_set:
            continue
        cid = record['pid']
        score = record['score']
        subqid_cid_dict[subqid].append((cid, score))
        qidpid_score_dict[(subqid, cid)] = score

    for record in whole_prop_records:
        qid = str(record['qid'])
        if qid not in qid_subqid_dict:
            continue
        pid = record['pid']
        score = record['score']
        qid_pid_dict[qid].append((pid, score))
        qidpid_score_dict[(qid, pid)] = score
    
    for record in multi_prop_records:
        subqid = str(record['qid'])
        if subqid not in subqid_set:
            continue
        pid = record['pid']
        score = record['score']
        subqid_pid_dict[subqid].append((pid, score))
        qidpid_score_dict[(subqid, pid)] = score
    
    # for full-rank: cand_passages will be the union of passages corresponding to queries and sub-queries
    # for re-rank: cand_passages is exactly the results from the proposition-level retrieval
    qid_cand_passages = defaultdict(list)
    
    # threshold
    # qid_cid_dict: {qid: [(cid, score)]}
    for qid, cid_scores in tqdm(qid_cid_dict.items()):
        qid_cid_dict[qid] = sorted(cid_scores, key=lambda k: k[1], reverse=True)[:threshold]
        qid_cand_passages[qid].extend([_cid for _cid, _ in qid_cid_dict[qid]])
    
    # subqid_cid_dict: {subqid: [(cid, score)]}
    for qid, cid_scores in tqdm(subqid_cid_dict.items()):
        subqid_cid_dict[qid] = sorted(cid_scores, key=lambda k: k[1], reverse=True)[:threshold]
        if not rerank:
            if 'condqa' in args.retrieval_dir:
                origin_qid = qid.split('%')[0]
            else:
                origin_qid = qid.split('#')[0]
            qid_cand_passages[origin_qid].extend([_cid for _cid, _ in subqid_cid_dict[qid]])

    # qid_pid_dict: {qid: [(pid, score)]}
    for qid, pid_scores in tqdm(qid_pid_dict.items()):
        qid_pid_dict[qid] = sorted(pid_scores, key=lambda k: k[1], reverse=True)[:threshold]
        qid_cand_passages[qid].extend(['-'.join(_pid.split('-')[:-1]) for _pid, _ in qid_pid_dict[qid]])
    
    # subqid_pid_dict: {subqid: [(pid, score)]}
    for qid, pid_scores in tqdm(subqid_pid_dict.items()):
        subqid_pid_dict[qid] = sorted(pid_scores, key=lambda k: k[1], reverse=True)[:threshold]
        if not rerank:
            if 'condqa' in args.retrieval_dir:
                origin_qid = qid.split('%')[0]
            else:
                origin_qid = qid.split('#')[0]
            qid_cand_passages[origin_qid].extend(['-'.join(_pid.split('-')[:-1]) for _pid, _ in subqid_pid_dict[qid]])
    
    for k, v in tqdm(qid_cand_passages.items(), desc='remove duplicate'):
        qid_cand_passages[k] = list(set(v))
        
    for record in tqdm(add_query_records):
        qid = str(record['qid'])
        # this can be chunk or prop
        pid = record['pid']

        if qid not in qid_subqid_dict:
            if 'condqa' in args.retrieval_dir:
                origin_qid = qid.split('%')[0]
            else:
                origin_qid = qid.split('#')[0]
        else:
            origin_qid = qid
        
        if pid in qid_cand_passages[origin_qid] or '-'.join(pid.split('-')[:-1]) in qid_cand_passages[origin_qid]:
            score = record['score']
            qidpid_score_dict[(qid, pid)] = score

    del add_query_records
    
    # scores between whole query and chunks
    qid_cid_score_dict = dict()

    for (qid, cids) in tqdm(qid_cand_passages.items(), desc='Calculate the similarity between whole queries and chunks'):
        for _cid in cids:
            print(qid, cids)
            qid_cid_score_dict[(qid, _cid)] = qidpid_score_dict[(qid, _cid)]

    # scores between subquery and chunks
    mean_qid_cid_score_dict = dict()
    for (qid, cids) in tqdm(qid_cand_passages.items(), desc='Calculate the similarity between subqueries and chunks'):
        for _cid in cids:
            _qid_size = len(qid_subqid_dict[qid])
            tmp_arr = np.zeros(_qid_size)
            for _qid_idx in range(_qid_size):
                tmp_arr[_qid_idx] = qidpid_score_dict[(qid_subqid_dict[qid][_qid_idx], _cid)]
            mean_qid_cid_score_dict[(qid, _cid)] = np.mean(tmp_arr).item()
    
    # scores between whole query and propositions
    qid_max_cid_score_dict = dict()
    for (qid, cids) in tqdm(qid_cand_passages.items(), desc='Calculate the similarity between whole queries and propositions'):
        for _cid in cids:
            if len(passage2prop[_cid]) > 0:
                qid_max_cid_score_dict[(qid, _cid)] = max([qidpid_score_dict[(qid, _prop_id)] for _prop_id in passage2prop[_cid]])
            else:
                qid_max_cid_score_dict[(qid, _cid)] = 0

    # scores between subquery and propositons
    conv_qid_cid_score_dict = dict()
    for (qid, cids) in tqdm(qid_cand_passages.items(), desc='Calculate the similarity between subqueries and propositions'):
        for _cid in cids:
            _qid_size = len(qid_subqid_dict[qid])
            _cid_size = len(passage2prop[_cid])

            if _cid_size == 0:
                conv_qid_cid_score_dict[(qid, _cid)] = 0
                continue
            
            tmp_arr = np.zeros((_qid_size, _cid_size))
            for _qid_idx in range(_qid_size):
                for _cid_idx in range(_cid_size):
                    tmp_arr[_qid_idx, _cid_idx] = qidpid_score_dict[(qid_subqid_dict[qid][_qid_idx], passage2prop[_cid][_cid_idx])]
            
            conv_qid_cid_score_dict[(qid, _cid)] = np.mean(np.max(tmp_arr, axis=1)).item()

    assert len(qid_max_cid_score_dict) == len(conv_qid_cid_score_dict), "The number of retrieved (query_id, passage_id) between queries and subqueries should be the same."
    
    # calculate the rank
    qid_cid_score_pair_dict = defaultdict(list)
    mean_qid_cid_score_pair_dict = defaultdict(list)
    qid_max_cid_score_pair_dict = defaultdict(list)
    conv_qid_cid_score_pair_dict = defaultdict(list)

    for (_qid, _cid), _score in qid_cid_score_dict.items():
        qid_cid_score_pair_dict[_qid].append((_cid, _score))

    for (_qid, _cid), _score in mean_qid_cid_score_dict.items():
        mean_qid_cid_score_pair_dict[_qid].append((_cid, _score))

    for (_qid, _cid), _score in qid_max_cid_score_dict.items():
        qid_max_cid_score_pair_dict[_qid].append((_cid, _score))

    for (_qid, _cid), _score in conv_qid_cid_score_dict.items():
        conv_qid_cid_score_pair_dict[_qid].append((_cid, _score))

    # sorted cid list for propostion list {qid: [cid, ...]}
    qid_cid_sort_list = {}
    mean_qid_cid_sort_list = {}
    qid_max_cid_sort_list = {}
    conv_qid_cid_sort_list = {}

    for _qid, _list in qid_cid_score_pair_dict.items():
        tmp_list = sorted(_list, key=lambda x: x[1], reverse=True)
        qid_cid_sort_list[_qid] = [_x[0] for _x in tmp_list]

    for _qid, _list in mean_qid_cid_score_pair_dict.items():
        tmp_list = sorted(_list, key=lambda x: x[1], reverse=True)
        mean_qid_cid_sort_list[_qid] = [_x[0] for _x in tmp_list]

    for _qid, _list in qid_max_cid_score_pair_dict.items():
        tmp_list = sorted(_list, key=lambda x: x[1], reverse=True)
        qid_max_cid_sort_list[_qid] = [_x[0] for _x in tmp_list]

    for _qid, _list in conv_qid_cid_score_pair_dict.items():
        tmp_list = sorted(_list, key=lambda x: x[1], reverse=True)
        conv_qid_cid_sort_list[_qid] = [_x[0] for _x in tmp_list]


    os.makedirs(os.path.join(args.retrieval_dir, 'rrf'), exist_ok=True)
    # creat diffent combination of rrf
    # we have to include
    # save mean_qid_cid_sort_list
    save_results(mean_qid_cid_score_pair_dict, os.path.join(args.retrieval_dir, 'rrf', 'mean.tsv'))

    # save conv_qid_cid_sort_list
    save_results(conv_qid_cid_score_pair_dict, os.path.join(args.retrieval_dir, 'rrf', 'conv.tsv'))


    # calculate and save (qid_max_cid_sort_list, conv_qid_cid_sort_list)
    rrf_qid_cid_max_dict = defaultdict(list)

    for _qid in tqdm(list(qid_cid_sort_list.keys()), desc='calculate rrf'):
        _cid_list = qid_cid_sort_list[_qid]
        _max_cid_list = qid_max_cid_sort_list[_qid]

        for _cid in _cid_list:
            _rank = _cid_list.index(_cid)
            _max_rank = _max_cid_list.index(_cid)
            _score = 1. / (1 + _rank) + 1. / (1 + _max_rank)
            rrf_qid_cid_max_dict[_qid].append((_cid, _score))

    save_results(rrf_qid_cid_max_dict, os.path.join(args.retrieval_dir, 'rrf', 'qid_cid_max.tsv'))

    # calculate and save (qid_max_cid_sort_list, conv_qid_cid_sort_list)
    rrf_qid_cid_conv_dict = defaultdict(list)

    for _qid in tqdm(list(qid_cid_sort_list.keys()), desc='calculate rrf'):
        _cid_list = qid_cid_sort_list[_qid]
        _conv_cid_list = conv_qid_cid_sort_list[_qid]

        for _cid in _cid_list:
            _rank = _cid_list.index(_cid)
            _conv_rank = _conv_cid_list.index(_cid)
            _score = 1. / (1 + _rank) + 1. / (1 + _conv_rank)
            rrf_qid_cid_conv_dict[_qid].append((_cid, _score))

    save_results(rrf_qid_cid_conv_dict, os.path.join(args.retrieval_dir, 'rrf', 'qid_cid_conv.tsv'))


    # calculate and save (qid_max_cid_sort_list, conv_qid_cid_sort_list)
    rrf_qid_max_conv_dict = defaultdict(list)

    for _qid in tqdm(list(qid_max_cid_sort_list.keys()), desc='calculate rrf'):
        _cid_list = qid_max_cid_sort_list[_qid]
        _conv_cid_list = conv_qid_cid_sort_list[_qid]

        for _cid in _cid_list:
            _max_rank = _cid_list.index(_cid)
            _conv_rank = _conv_cid_list.index(_cid)
            _score = 1. / (1 + _max_rank) + 1. / (1 + _conv_rank)
            rrf_qid_max_conv_dict[_qid].append((_cid, _score))
    
    save_results(rrf_qid_max_conv_dict, os.path.join(args.retrieval_dir, 'rrf', 'max_conv.tsv'))

    # calculate and save (qid_max_cid_sort_list, conv_qid_cid_sort_list)
    rrf_qid_max_conv_dict = defaultdict(list)

    for _qid in tqdm(list(qid_max_cid_sort_list.keys()), desc='calculate rrf'):
        _cid_list = qid_max_cid_sort_list[_qid]
        _conv_cid_list = conv_qid_cid_sort_list[_qid]

        for _cid in _cid_list:
            _max_rank = _cid_list.index(_cid)
            _conv_rank = _conv_cid_list.index(_cid)
            _score = 1. / (1 + _max_rank) + 1. / (1 + _conv_rank)
            rrf_qid_max_conv_dict[_qid].append((_cid, _score))
    
    save_results(rrf_qid_max_conv_dict, os.path.join(args.retrieval_dir, 'rrf', 'max_conv.tsv'))

    # calculate and save (qid_cid_sort_list, qid_max_cid_sort_list, conv_qid_cid_sort_list)
    rrf_qid_cid_max_conv_dict = defaultdict(list)

    for _qid in tqdm(list(qid_cid_sort_list.keys()), desc='calculate rrf'):
        _cid_list = qid_cid_sort_list[_qid]
        _max_cid_list = qid_max_cid_sort_list[_qid]
        _conv_cid_list = conv_qid_cid_sort_list[_qid]

        for _cid in _cid_list:
            _rank = _cid_list.index(_cid)
            _max_rank = _max_cid_list.index(_cid)
            _conv_rank = _conv_cid_list.index(_cid)
            _score = 1. / (1 + _rank) + 1. / (1 + _max_rank) + 1. / (1 + _conv_rank)
            rrf_qid_cid_max_conv_dict[_qid].append((_cid, _score))

    save_results(rrf_qid_cid_max_conv_dict, os.path.join(args.retrieval_dir, 'rrf', 'qid_cid_max_conv.tsv'))


    # calculate and save (qid_cid_sort_list, mean_qid_cid_sort_list, qid_max_cid_sort_list, conv_qid_cid_sort_list)
    rrf_mean_qid_cid_max_conv_dict = defaultdict(list)

    for _qid in tqdm(list(qid_cid_sort_list.keys()), desc='calculate rrf'):
        _cid_list = qid_cid_sort_list[_qid]
        _mean_cid_list = mean_qid_cid_sort_list[_qid]
        _max_cid_list = qid_max_cid_sort_list[_qid]
        _conv_cid_list = conv_qid_cid_sort_list[_qid]

        for _cid in _cid_list:
            _rank = _cid_list.index(_cid)
            _mean_rank = _mean_cid_list.index(_cid)
            _max_rank = _max_cid_list.index(_cid)
            _conv_rank = _conv_cid_list.index(_cid)
            _score = 1. / (1 + _rank) + 1. / (1 + _mean_rank) + 1. / (1 + _max_rank) + 1. / (1 + _conv_rank)
            rrf_mean_qid_cid_max_conv_dict[_qid].append((_cid, _score))

    save_results(rrf_mean_qid_cid_max_conv_dict, os.path.join(args.retrieval_dir, 'rrf', 'qid_cid_mean_max_conv.tsv'))
    

def run(args):
    retrieval_dir = args.retrieval_dir

    whole_prop_file = os.path.join(retrieval_dir, args.whole_prop_path)
    multi_prop_file = os.path.join(retrieval_dir, args.multi_prop_path)
    whole_chunk_file = os.path.join(retrieval_dir, args.whole_chunk_path)
    multi_chunk_file = os.path.join(retrieval_dir, args.multi_chunk_path)
    add_query_file = os.path.join(retrieval_dir, args.add_query_file)

    whole_prop_records = get_records(whole_prop_file, None, threshold=args.threshold)
    multi_prop_records = get_records(multi_prop_file, None, threshold=args.threshold)
    whole_chunk_records = get_records(whole_chunk_file, None, threshold=args.threshold)
    multi_chunk_records = get_records(multi_chunk_file, None, threshold=args.threshold)

    print('Loading add...')
    add_query_records = pd.read_csv(add_query_file, sep='\t').to_dict('records')
    print('Done')
    
    threshold = 200
    rerank = False
    conv_by_encoder(args,
                    whole_chunk_records,
                    whole_prop_records,
                    multi_chunk_records,
                    multi_prop_records,
                    add_query_records,
                    rerank,
                    threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder model name or path', required=True)
    parser.add_argument('--threshold', type=int, default=-1)
    
    parser.add_argument('--sub_query_jsonl', type=str, required=True)
    parser.add_argument('--retrieval_dir', type=str, required=True)
    parser.add_argument('--whole_prop_path', type=str, required=True)
    parser.add_argument('--multi_prop_path', type=str, required=True)
    parser.add_argument('--whole_chunk_path', type=str, required=True)
    parser.add_argument('--multi_chunk_path', type=str, required=True)
    parser.add_argument('--add_query_file', type=str, required=True)
    parser.add_argument('--passage2prop', type=str, required=True)


    args = parser.parse_args()

    global passage2prop
    with open(args.passage2prop, 'rb') as f:
        passage2prop = pickle.load(f)

    global qid_subqid_dict
    qid_subqid_dict = defaultdict(list)
    with open(os.path.join(args.sub_query_jsonl), 'r') as f:
        subqid_content_list = [json.loads(line) for line in f.readlines()]
        for line in subqid_content_list:
            qid = line['id']
            if 'condqa' in args.retrieval_dir:
                origin_qid = qid.split('%')[0]
            else:
                origin_qid = qid.split('#')[0]
            qid_subqid_dict[origin_qid].append(qid)
    
    print("#query:", len(qid_subqid_dict))
    run(args)
