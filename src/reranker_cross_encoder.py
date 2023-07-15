from sentence_transformers import SentenceTransformer, CrossEncoder, util
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
cross_encoder = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

bm25_folderpath = DATA_PATH + 'retriever_output/bm25/'
semantic_pool_folderpath = DATA_PATH + 'retriever_output/semantic_pool/'
candidate_pool_folderpath = DATA_PATH + 'retriever_output/candidate_pool/'
optimized_candidate_pool_folderpath = DATA_PATH + 'retriever_output/optimized_candidate_pool/'

bm25_output_folderpath = DATA_PATH + 'reranker_output/bm25/'
semantic_pool_output_folderpath = DATA_PATH + 'reranker_output/semantic_pool/'
candidate_pool_output_folderpath = DATA_PATH + 'reranker_output/candidate_pool/'
optimized_candidate_pool_output_folderpath = DATA_PATH + 'reranker_output/optimized_candidate_pool/'

LOG_FILE = DATA_PATH + f'logs/retriever_data_generator_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

def write_to_pickle(filename, data):

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_from_pickle(filename):

    with open(filename, 'rb') as f:         
        data = pickle.load(f)

    return data


def cross_reranker(filename):

    retriever_data = read_from_pickle(filename)

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[idx]] for idx in retriever_data]
    cross_scores = cross_encoder.predict(cross_inp)

    hits = []
    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits.append(dict())
        hits[idx]['id'] = retriever_data[idx]
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'][1], reverse=True)
    
    crossencoder_passage_id_list = [hit['id'] for hit in hits]

    return crossencoder_passage_id_list


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

        top_k = 100                        

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

            # bm25_reranker_output = cross_reranker(bm25_folderpath + query_updated +'.pickle')
            # write_to_pickle(bm25_output_folderpath + query_updated +'.pickle', bm25_reranker_output)

            # semantic_reranker_output = cross_reranker(semantic_pool_folderpath + query_updated +'.pickle')
            # write_to_pickle(semantic_pool_output_folderpath + query_updated +'.pickle', semantic_reranker_output)

            # candidate_pool_reranker_output = cross_reranker(candidate_pool_folderpath + query_updated +'.pickle')
            # write_to_pickle(candidate_pool_output_folderpath + query_updated +'.pickle', candidate_pool_reranker_output)

            optimzed_candidate_pool_reranker_output = cross_reranker(optimized_candidate_pool_folderpath + query_updated +'.pickle')
            write_to_pickle(optimized_candidate_pool_output_folderpath + query_updated +'.pickle', optimzed_candidate_pool_reranker_output)

            logging.info(f"Finished {query} ... ")
    
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

    logging.info("Finished retriever data generator ... ")