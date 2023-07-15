import os
import numpy as np
import pandas as pd
import logging
import pickle
import traceback

DATA_PATH = '/usr/src/app/data/'
optimized_candidate_pool_folderpath = DATA_PATH + 'retriever_output/optimized_candidate_pool/'

LOG_FILE = DATA_PATH + f'logs/retriever_data_generator_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)


def write_to_pickle(filename, data):

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':

    logging.info("Started optimized candidate pool generator ....")
    try:

        xlm_df = pd.read_pickle(DATA_PATH+'final_keywords_dataframe_cdd.pkl')

        passage_id_list = []
        for idx, row in xlm_df.iterrows():

            passage_id = row['id']
            passage_id_list.append(passage_id)

        logging.info(f"Passages: {len(passage_id_list)}")

        query_dataframe_folderpath = DATA_PATH+'dataframes/query_dataframes_updated_nc_large_cdd/'
        target_queries = ['Kryptologie', 'Defense', 'Cyber Attack', 'Data Centric Warfare', 'unbemannte Wirksysteme', ]

        for query_df_filename in os.listdir(query_dataframe_folderpath):

            query_df = pd.read_pickle(query_dataframe_folderpath+query_df_filename)
            query = query_df['query'].values[0]
            query_updated = query.lower().replace(' ', '_')

            if query in target_queries:
                continue

            query_doc_id_list = list(query_df.page_id.values)
            xlm_df = pd.read_pickle(DATA_PATH+'final_keywords_dataframe_cdd.pkl')

            candidate_pool_output = [passage_id_list.index(val) for val in query_doc_id_list]
            write_to_pickle(optimized_candidate_pool_folderpath + query_updated +'.pickle', candidate_pool_output)
                
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
            

    logging.info("Finished optimized candidate pool generator ....")
