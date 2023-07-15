import os
import string
import json
import pickle
import numpy as np
import logging
import traceback

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/usr/src/app/data/'
pickle_read_filepath = DATA_PATH+'final_keywords_dataframe.pkl'
pickle_write_filepath = DATA_PATH+'final_keywords_dataframe_cdd.pkl'

keyword_list_filepath = DATA_PATH + f'keyword_features_pkl/'
noun_chunk_list_filepath = DATA_PATH + f'noun_chunk_features_pkl/'

LOG_FILE = DATA_PATH + f'logs/redisdb_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                encoding='utf-8', mode='a+')],
                    level=logging.INFO)

def read_document_data(filepath):

    try:
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)

        data_dict = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logging.error(traceback.print_exc())
        print(e)

    return data_dict

def load_processed_doc_list():

    doc_name_list = []
    for filename in os.listdir(noun_chunk_list_filepath):
        doc_name_list.append(filename.split('.pkl')[0])

    return doc_name_list

def get_text_tokens(filepath):

    try:
        with open(filepath, 'rb') as f:
            text_tokens = pickle.load(f)

    except Exception as e:
        logging.error(traceback.print_exc())
        print(e)

    return text_tokens

def get_keyword_noun_chunk_df(processed_doc_list):

    meta_data = []
    for doc_id in processed_doc_list:

        text_tokens = get_text_tokens(noun_chunk_list_filepath + doc_id + '.pkl')
        keywords = read_document_data(keyword_list_filepath + doc_id + '.pkl')

        meta_data.append({
                        'id': doc_id,
                        'text_tokens': text_tokens,
                        'keywords': keywords
        })

    kw_df = pd.DataFrame(meta_data)

    return kw_df


if __name__ == '__main__':

    logging.info("Started load keywords from json ....")

    try:
        df_xlm = pd.read_pickle(pickle_read_filepath)
        processed_doc_list = load_processed_doc_list()

        df_xlm_cdd = df_xlm[df_xlm['id'].isin(processed_doc_list)]
        df_xlm_cdd.drop('text_tokens', axis=1, inplace=True)
        df_xlm_cdd.drop('keywords', axis=1, inplace=True)

        df_kw_tt = get_keyword_noun_chunk_df(processed_doc_list)

        df_xlm_cdd = pd.concat([df_xlm_cdd.set_index('id'), df_kw_tt.set_index('id')], axis=1, join='inner').reset_index()

        # column_names = df_xlm.columns.values
        # column_names[1] = 'keywords_pickle'
        # df_xlm.columns = column_names

        # df_xlm.drop('keywords_pickle', axis=1, inplace=True)

        df_xlm_cdd.to_pickle(pickle_write_filepath)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.print_exc())

    logging.info("Finished load keywords from json ....")