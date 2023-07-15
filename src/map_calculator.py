import os
import numpy as np
import pandas as pd
import logging
import pickle
import traceback

DATA_PATH = '/usr/src/app/data/'

query_dataframe_folderpath = DATA_PATH+'dataframes/query_dataframes_updated_nc_small_cdd/'

bm25_output_folderpath = DATA_PATH + 'reranker_output/bm25/'
semantic_pool_output_folderpath = DATA_PATH + 'reranker_output/semantic_pool/'
candidate_pool_output_folderpath = DATA_PATH + 'reranker_output/candidate_pool/'
optimized_candidate_pool_output_folderpath = DATA_PATH + 'reranker_output/optimized_candidate_pool/'

# bm25_map_folderpath = DATA_PATH + 'map_output/bm25/'
# semantic_pool_map_folderpath = DATA_PATH + 'map_output/semantic_pool/'
# candidate_pool_map_folderpath = DATA_PATH + 'map_output/candidate_pool/'

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

def get_positive_doc_ids(df):

    doc_ids = []
    for idx, row in df.iterrows():

        label = row['label']
        doc_id = row['page_id']

        if label == 1:
            doc_ids.append(doc_id)

    return doc_ids

def calculate_ap(query_doc_ids, filename):

    reranker_data = read_from_pickle(filename)
    reranker_doc_ids = [passage_id_list[idx] for idx in reranker_data]

    pos_doc_cnt = 0
    query_avg_exp = []
    for idx, doc_id in enumerate(reranker_doc_ids):
        if doc_id in query_doc_ids:
            pos_doc_cnt += 1
            query_avg_exp.append(round((pos_doc_cnt/(idx+1)), 3))

    if len(query_avg_exp) == 0:
        ap_query = 0
    else:
        ap_query = round(sum(query_avg_exp)/len(query_avg_exp), 3)

    return ap_query

def calculate_map(query_ap_list):

    map_data = round(sum(query_ap_list)/len(query_ap_list), 3)
    return map_data

if __name__ == '__main__':

    logging.info("Starting retriever data generator ... ")

    try:
        xlm_df = pd.read_pickle(DATA_PATH+'final_keywords_dataframe_cdd.pkl')
        target_queries = ['Kryptologie', 'Defense', 'Cyber Attack', 'Data Centric Warfare', 'unbemannte Wirksysteme', ]

        passage_id_list = []
        passages = []
        for idx, row in xlm_df.iterrows():

            passage_id = row['id']
            passage = row['text']

            passage_id_list.append(passage_id)
            passages.append(passage)

        logging.info(f"Passages: {len(passages)}")

        bm25_map_list = []
        semantic_pool_map_list = []
        candidate_pool_map_list = []
        optimized_candidate_pool_map_list = []

        for query_df_filename in os.listdir(query_dataframe_folderpath):

            query_df = pd.read_pickle(query_dataframe_folderpath+query_df_filename)
            query = query_df['query'].values[0]

            if query in target_queries:
                continue

            query_updated = query.lower().replace(' ', '_')
            query_doc_ids = get_positive_doc_ids(query_df)

            # bm25_map_output = calculate_ap(query_doc_ids, bm25_output_folderpath + query_updated +'.pickle')

            # semantic_map_output = calculate_ap(query_doc_ids, semantic_pool_output_folderpath + query_updated +'.pickle')

            # candidate_pool_map_output = calculate_ap(query_doc_ids, candidate_pool_output_folderpath + query_updated +'.pickle')

            optimized_candidate_pool_map_output = calculate_ap(query_doc_ids, optimized_candidate_pool_output_folderpath + query_updated +'.pickle')

            # bm25_map_list.append(bm25_map_output)
            # semantic_pool_map_list.append(semantic_map_output)
            # candidate_pool_map_list.append(candidate_pool_map_output)
            optimized_candidate_pool_map_list.append(optimized_candidate_pool_map_output)

            logging.info(f"Finished {query} ... ")

        # bm25_map = calculate_map(bm25_map_list)
        # semantic_pool_map = calculate_map(semantic_pool_map_list)
        # candidate_pool_map = calculate_map(candidate_pool_map_list)
        optimzed_candidate_pool_map = calculate_map(optimized_candidate_pool_map_list)

        # map_output_data = [bm25_map, semantic_pool_map, candidate_pool_map]
        map_output_data = [optimzed_candidate_pool_map]
        write_to_pickle(DATA_PATH+'map_output_data.pickle', map_output_data)
    
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

    logging.info("Finished retriever data generator ... ")