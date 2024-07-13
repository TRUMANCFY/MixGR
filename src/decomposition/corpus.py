from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import os

model_name = "chentong00/propositionizer-wiki-flan-t5-large"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

import spacy
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
nlp.enable_pipe("senter")


def append_to_result(result, file):
    if os.path.exists(file):
        append_write = 'a'
    else:
        append_write = 'w'

    if isinstance(result, dict):
        result = json.dumps(result)

    with open(file, append_write) as w:
        w.write(result + '\n')

def to_chunks(text, max_len=128):
    doc = nlp(text)
    sent_text = [sent.text for sent in doc.sents]
    sent_len = [len(sent) for sent in doc.sents]
    chunk_text = []
    last_sent_id, last_len = 0, 0
    # if adding the next sentence to the current group does not exceed the max_len, add it
    for i in range(len(sent_text)):
        if i + 1 >= len(sent_text) or last_len + sent_len[i + 1] > max_len:
            chunk_text.append(" ".join(sent_text[last_sent_id:i + 1]))
            last_sent_id = i + 1
            last_len = 0
        else:
            last_len += sent_len[i + 1]
    print(f"[INFO] Split text into {len(chunk_text)} chunks.")
    return chunk_text

def generate_chunk(args, lines):
    text_map = []
    res_chunks = []
    pure_chunks = []
    for _passage_idx, line in tqdm(enumerate(lines), total=len(lines)):
        chunks = line['contents'].split('\n')
        title = chunks[0]
        content = chunks[1].replace('\\n', ' ')
        section = ""

        chunks = to_chunks(content, max_len=args.max_len)
        pure_chunks.append(chunks)

        for chunk in chunks:
            input_text = f"Title: {title}. Section: {section}. Content: {chunk}"
            res_chunks.append(input_text)
            text_map.append(_passage_idx)
    
    return pure_chunks, res_chunks, text_map

def parse_contents(args, chunks):
    batch_size = 64
    num_batch = (len(chunks) // batch_size) + int(len(chunks) % batch_size > 0)
    
    prop_list = []
    for batch_idx in tqdm(range(num_batch)):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        ins_inputs = chunks[start_idx:end_idx]
        ins_input_ids = tokenizer(ins_inputs,
                            return_tensors="pt",
                            padding="max_length",
                            max_length=512,
                            truncation=True).input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(ins_input_ids, max_new_tokens=512).cpu().detach()
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        prop_list.extend(output_text)
    
    return prop_list


def main_prop(args):
    # input: chunk file
    with open(args.parse_file, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    chunks = []
    raw_chunks = []
    chunk_ids = []
    id_title_map = {}

    for line in lines:
        text_splits = line['contents'].split('\n')
        title = text_splits[0]
        section = ""
        chunk = text_splits[1].replace('\\n', ' ') if len(text_splits) == 2 else ""

        input_text = f"Title: {title}. Section: {section}. Content: {chunk}"
        chunks.append(input_text)
        raw_chunks.append(chunk)
        chunk_ids.append(line['id'])
        id_title_map[line['id']] = title

    prop_list = parse_contents(args, chunks)
    prop_lines = []

    for _prop_str_idx in range(len(prop_list)):
        chunk_id = chunk_ids[_prop_str_idx]
        prop_str = prop_list[_prop_str_idx]
        chunk = chunks[_prop_str_idx]
        raw_chunk = raw_chunks[_prop_str_idx]

        if raw_chunk == "":
            local_prop_list = [""]
        else:
            try:
                local_prop_list = json.loads(prop_str)
            except:
                print(f"[ERROR] Failed to parse: {prop_str} {chunk_id}")
                local_prop_list = []

        for _prop_idx, _prop in enumerate(local_prop_list):
            prop_lines.append({
                'id': chunk_id + '-' + str(_prop_idx),
                'contents':  id_title_map[chunk_id] + '\n' + str(_prop).replace('\n', '\\n'),
            })
   
    write_lines = '\n'.join([json.dumps(line) for line in prop_lines])
    save_path = args.parse_file.replace('corpus.chunk', 'corpus.prop')
    append_to_result(write_lines, save_path)


def main(args):
    with open(args.parse_file, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    # pure_chunks: list of list of str
    # res_chunks: list of str
    pure_chunks, res_chunks, text_map = generate_chunk(args, lines)

    # generate chunk lines
    chunk_lines = []
    for chunks, line in zip(pure_chunks, lines):
        title = line['contents'].split('\n')[0]
        for _chunk_idx, chunk in enumerate(chunks):
            chunk_lines.append({
                'id': line['id'] + '-' + str(_chunk_idx),
                'contents': title + '\n' + chunk.replace('\n', '\\n'),
            })
    
    if args.chunk_file is not None:
        save_chunk_file = os.path.join(os.path.dirname(args.parse_file), args.chunk_file)
    else:
        save_chunk_file = os.path.join(os.path.dirname(args.parse_file), 'corpus.chunk.jsonl')
    write_lines = '\n'.join([json.dumps(line) for line in chunk_lines])
    append_to_result(write_lines, save_chunk_file)

    args.parse_file = save_chunk_file
    main_prop(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse_file', required=True, type=str)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--chunk_exist', action='store_true', default=False)
    parser.add_argument('--chunk_file', type=str, default=None)

    args = parser.parse_args()

    if args.chunk_exist:
        main_prop(args)
    else:
        main(args)