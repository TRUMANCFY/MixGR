#!/bin/bash

ENCODER=$1

declare -A RETRIEVER

RETRIEVER["ance"]=castorini/ance-dpr-context-multi
RETRIEVER["contriever"]=facebook/contriever
RETRIEVER["simcse"]=princeton-nlp/unsup-simcse-bert-base-uncased
RETRIEVER["dpr"]=facebook/dpr-ctx_encoder-multiset-base
RETRIEVER["tasb"]=sentence-transformers/msmarco-distilbert-base-tas-b
RETRIEVER["gtr"]=sentence-transformers/gtr-t5-base

ENCODER_MODEL="${RETRIEVER["$ENCODER"]}"
echo $ENCODER_MODEL

python -m pyserini.encode \
input   --corpus $PROP_JSONL \
        --fields title text \
        --delimiter "\n" \
output  --embeddings $PROP_INDEX \
        --to-faiss \
encoder --encoder $ENCODER_MODEL \
        --fields title text \
        --batch 16 \
        --fp16 \
        --device cuda:0