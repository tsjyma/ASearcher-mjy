#!/bin/bash
set -ex

WIKI2018_WORK_DIR=/root/autodl-tmp/hub/datasets--inclusionAI--ASearcher-Local-Knowledge/snapshots/16091d9f248bd7147225f17bf334a5b55669de1e/

index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
corpus_file=$WIKI2018_WORK_DIR/wiki_corpus_small.jsonl
pages_file=None # $WIKI2018_WORK_DIR/wiki_webpages.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

CUDA_VISIBLE_DEVICES=2,3 python tools/local_retrieval_server.py --index_path $index_file \
                                        --corpus_path $corpus_file \
                                        --pages_path $pages_file \
                                        --topk 3 \
                                        --retriever_name $retriever_name \
                                        --retriever_model $retriever_path \
                                        --faiss_gpu --port $1 \
                                        --save-address-to $2
