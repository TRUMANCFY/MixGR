ENCODER=$1

declare -A RETRIEVER

RETRIEVER["simcse"]=princeton-nlp/unsup-simcse-bert-base-uncased
RETRIEVER["contriever"]=facebook/contriever
RETRIEVER["dpr"]=facebook/dpr-question_encoder-multiset-base
RETRIEVER["ance"]=castorini/ance-dpr-question-multi
RETRIEVER["tasb"]=sentence-transformers/msmarco-distilbert-base-tas-b
RETRIEVER["gtr"]=sentence-transformers/gtr-t5-base

ENCODER_MODEL="${RETRIEVER["$ENCODER"]}"
echo $ENCODER_MODEL

python -m pyserini.search.faiss \
--encoder $ENCODER_MODEL \
--index $PROP_INDEX \
--topics $SUBQUERY_PATH \
--output $SUBQUERY_PROP_SEARCH_PATH \
--batch-size 64 --threads 32 \
--hits 500