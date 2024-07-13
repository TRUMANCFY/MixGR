import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
import random
import argparse
random.seed(9001)

# Model
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
import torch

def append_to_result(result, file):
    if os.path.exists(file):
        append_write = 'a'
    else:
        append_write = 'w'

    if isinstance(result, dict):
        result = json.dumps(result)

    with open(file, append_write) as w:
        w.write(result + '\n')

def main(args, dataset):
    model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    device = "cuda"
    prop_tokenizer = AutoTokenizer.from_pretrained(model_name)
    prop_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    prop_model.eval()

    SEGMENT5_PROMPT = "Title: . Section: . Content: {}"

    batch_size = args.batch_size
    num_batch = (len(dataset) // batch_size) + int(len(dataset) % batch_size > 0)
    
    for batch_idx in tqdm(range(num_batch)):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        lines = dataset[start_idx:end_idx]

        ins_inputs = [SEGMENT5_PROMPT.format(ins['input']) for ins in lines]
        ins_input_ids = prop_tokenizer(ins_inputs,
                                    return_tensors="pt",
                                    padding="max_length",
                                    max_length=512,
                                    truncation=True).input_ids.to(device)
        
        outputs = prop_model.generate(ins_input_ids, max_new_tokens=512).cpu().detach()

        output_text = prop_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for _line, _output_text in zip(lines, output_text):

            try:
                prop_list = json.loads(_output_text)
            except:
                prop_list = _output_text
            
            res = {
                'id': _line['id'],
                'input': _line['input'],
                'prop_generation': prop_list,
            }

            append_to_result(res, save_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        # different types of models
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    args = parser.parse_args()

    dataset = []
    with open(args.query_file, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
        for line in lines:
            dataset.append({
                'id': line['id'],
                'input': line['title'],
            })

    global save_path
    save_path = args.save_file

    main(args, dataset)
