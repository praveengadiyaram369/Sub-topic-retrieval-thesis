import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
from nltk.corpus import stopwords
import string
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import logging
import pickle
import traceback

DATA_PATH = '/usr/src/app/data/'
german_stop_words = stopwords.words('german')
bi_encoder = SentenceTransformer('LLukas22/all-MiniLM-L12-v2-embedding-all')
cross_encoder = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

bm25_folderpath = DATA_PATH + 'retriever_output/bm25/'
semantic_pool_folderpath = DATA_PATH + 'retriever_output/semantic_pool/'
candidate_pool_folderpath = DATA_PATH + 'retriever_output/candidate_pool/'


LOG_FILE = DATA_PATH + f'logs/retriever_data_generator_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS and token not in german_stop_words:
            tokenized_doc.append(token)
    return tokenized_doc

def lexical_search(query, top_k):

    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -top_k)[-top_k:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    bm25_passage_id_list = [hit['corpus_id'] for hit in bm25_hits]

    return bm25_passage_id_list

def semantic_search(query, top_k):

    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    semantic_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    semantic_hits = semantic_hits[0] 
    semantic_hits = sorted(semantic_hits, key=lambda x: x['score'], reverse=True)

    semantic_passage_id_list = [hit['corpus_id'] for hit in semantic_hits]

    return semantic_passage_id_list


def search(query):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    print("Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
#     question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'][1], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'][1], passages[hit['corpus_id']].replace("\n", " ")))

def write_to_pickle(filename, data):

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def create_pickle_files(passages):

    tokenized_corpus = []
    for passage in tqdm(passages):
        tokenized_corpus.append(bm25_tokenizer(passage))

    bm25 = BM25Okapi(tokenized_corpus)
    write_to_pickle(DATA_PATH+'bm25_pickle.pickle', bm25)

    logging.info("Finished bm25 saving to pickle ... ")

    corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

    write_to_pickle(DATA_PATH+'passage_embeddings_pickle.pickle', corpus_embeddings)

    logging.info("Finished corpus_embeddings saving to pickle ... ")


def read_pickle_files():

    bm25 = None
    corpus_embeddings = None

    with open(DATA_PATH+'bm25_pickle.pickle', 'rb') as file1:         
        bm25=pickle.load(file1)
        
    with open(DATA_PATH+'passage_embeddings_pickle.pickle', 'rb') as file2:         
        corpus_embeddings=pickle.load(file2)

    return bm25, corpus_embeddings


if __name__ == '__main__':

    logging.info("Starting retriever data generator ... ")

    try:
        xlm_df = pd.read_pickle(DATA_PATH+'final_keywords_dataframe_cdd.pkl')

        passage_id_list = []
        passages = []
        for idx, row in xlm_df.iterrows():

            passage_id = row['id']
            passage = row['text']

            passage_id_list.append(passage_id)
            passages.append(passage)

        logging.info(f"Passages: {len(passages)}")

        bi_encoder.max_seq_length = 512     
        top_k = 100                        
        bm25, corpus_embeddings = read_pickle_files()
        # create_pickle_files(passages)

        logging.info("Finished read pickles ... ")

        queries_list = ['Edge computing',
                        'Unbemannte Landsysteme',
                        'Kommunikationsnetze',
                        'Methode Architektur',
                        'IT-Standards',
                        'Mixed Reality',
                        'Visualisierung',
                        'Architekturanalyse',
                        'Robotik',
                        'Waffen Systeme',
                        'Satellitenkommunikation',
                        'Militärische Kommunikation',
                        'Schutz von unbemannten Systemen',
                        'militärische Entscheidungsfindung',
                        'Quantentechnologie',
                        'Big Data, KI für Analyse',
                        'Wellenformen und -ausbreitung']

        for query in queries_list:
            query_updated = query.lower().replace(' ', '_')

            bm25_output = lexical_search(query, top_k)
            write_to_pickle(bm25_folderpath + query_updated +'.pickle', bm25_output)

            semantic_output = semantic_search(query, top_k)
            write_to_pickle(semantic_pool_folderpath + query_updated +'.pickle', semantic_output)

            candidate_pool_output = bm25_output[:50] + semantic_output[:50]
            write_to_pickle(candidate_pool_folderpath + query_updated +'.pickle', candidate_pool_output)

            logging.info(f"Finished {query} ... ")
    
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

    logging.info("Finished retriever data generator ... ")