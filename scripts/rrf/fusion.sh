#!/bin/bash

ENCODER=$1

python $ROOT_DIR/src/rrf/fusion.py \
--encoder $ENCODER \
--sub_query_jsonl $SUBQUERY_PATH \
--retrieval_dir $RETRIEVAL_DIR \ # the retrieval directory containing all retrieval-related results
--whole_chunk_path $QUERY_CHUNK_SEARCH_PATH \
--multi_chunk_path $SUBQUERY_CHUNK_SEARCH_PATH \
--whole_prop_path $QUERY_PROP_SEARCH_PATH \
--multi_prop_path $SUBQUERY_PROP_SEARCH_PATH \
--add_query_file $ADDITIIONAL_SEARCH_PATH \ # output file of union.py
--passage2prop $PASSAGE2PROP_PKL_PATH