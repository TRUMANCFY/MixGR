#!/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $CHUNK_JSONL_DIR \
  --index $BM25_INDEX_PATH \
  --generator DefaultLuceneDocumentGenerator \
  --storePositions --storeDocvectors --storeRaw
