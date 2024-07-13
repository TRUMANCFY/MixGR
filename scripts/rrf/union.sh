#!/bin/bash

ENCODER=$1

python $ROOT_DIR/src/rrf/union.py \
--encoder $ENCODER \
--prop_bm25_dir $PROP_BM25_INDEX_PATH \
--chunk_bm25_dir $CHUNK_BM25_INDEX_PATH \
--subquery_file $SUBQUERY_PATH \
--query_file $QUERY_PATH \
--passage2prop $PASSAGE2PROP_PKL_PATH \
--whole_prop_path $QUERY_CHUNK_SEARCH_PATH \
--multi_prop_path $QUERY_PROP_SEARCH_PATH \
--whole_chunk_path $SUBQUERY_CHUNK_SEARCH_PATH \
--multi_chunk_path $SUBQUERY_PROP_SEARCH_PATH