import os
import numpy as np
import pandas as pd
import logging
import traceback
import cleantext
import json

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

from sklearn.metrics.pairwise import cosine_similarity

import redis
import msgpack
import msgpack_numpy as m
m.patch()

import time
import warnings
warnings.filterwarnings('ignore')

from noun_chunk_extraction_xxx import *

DATA_PATH = '/usr/src/app/data/'
tf_model = None

# xlm_df = pd.read_pickle(DATA_PATH+'xlm_dataframe.pkl')
xlm_df = pd.read_pickle(DATA_PATH+'final_keywords_dataframe_cdd.pkl')
xlm_df = xlm_df.rename(columns={'id': 'page_id'})

LOG_FILE = DATA_PATH + f'logs/candidate_pool_generation_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

rdb = redis.StrictRedis(
    host='redis_cache',
    port=6379,
)

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_pool_vec(doc_vec_list, pool):
    
    doc_vec_list = get_modified_vectors(doc_vec_list)
    if pool == 'mean':
        return np.nanmean(doc_vec_list, axis=0)
    elif pool == 'max':
        return np.nanmax(doc_vec_list, axis=0)

def get_document_vec(text):
    
    return tf_model(text)['outputs'].numpy()[0].reshape(1, -1)

def get_sent_transformers_keywords_use(keywords, query_vec, max_keyword_cnt = 30):
    
    keywords = list(dict(keywords).keys())
    
    candidate_embeddings_keywords = [m.unpackb(rdb.get(kw)) for kw in keywords]
    # candidate_embeddings_keywords = [tf_model(kw)['outputs'].numpy()[0] for kw in keywords]
        
    query_distances = cosine_similarity([query_vec], candidate_embeddings_keywords)
    subtopic_keywords_dict = dict()
    for index in query_distances.argsort()[0][-max_keyword_cnt:]: 
        
        subtopic_keywords_dict[keywords[index]] = query_distances[0][index]
    
    subtopic_keywords_dict = sorted(subtopic_keywords_dict.items(), key=lambda x: x[1], reverse=True)

    return subtopic_keywords_dict

def get_candidate_pool(subtopic_keywords_list, lower_limit = 0.2, upper_limit = 0.4):
    
    candidate_pool = []
    for key, value in subtopic_keywords_list:
        
        if value > lower_limit and value < upper_limit:
            candidate_pool.append(key)
            
    return candidate_pool

def get_clean_text(text):
    
    clean_text = cleantext.clean(text,
            clean_all= False, # Execute all cleaning operations
            extra_spaces=True ,  # Remove extra white spaces 
            )
    
    clean_text = re.sub(r'http\S+', '', clean_text)
    p = re.compile(r'<.*?>')
    return p.sub('', clean_text)

def get_representation_vector(document, title):
    
    title_vec = get_document_vec(title)
    
    document_tokens = document.split()
    doc_len = len(document_tokens)
    doc_vecs = []
    
    doc_vecs.append(title_vec)
    
    if doc_len < 550:
        doc_vecs.append(get_document_vec(document))
    else:
        doc_parts = int(doc_len/500)
        for idx in range(doc_parts):
            if (idx+1)*500 >= doc_len:
                doc_temp = ' '.join(document_tokens[idx*500:])
            else:
                doc_temp = ' '.join(document_tokens[idx*500:(idx+1)*500])
                
            doc_vecs.append(get_document_vec(doc_temp))
        
    return get_pool_vec(get_modified_vectors(doc_vecs), pool='mean')

def get_sent_transformers_keywords(repr_vec, noun_chunks, max_keyword_cnt = 30):

    keywords_dict = dict()
    
    try:
        candidate_embeddings = [tf_model(nc)['outputs'].numpy()[0] for nc in noun_chunks]
        
        kw_distances = cosine_similarity([repr_vec], candidate_embeddings)
        
        data_insert_dict = dict()
        keywords_dict = dict()
        for index in kw_distances.argsort()[0][-max_keyword_cnt:]: 
            
            data_insert_dict[noun_chunks[index]] =  m.packb(candidate_embeddings[index])
                    
            keywords_dict[noun_chunks[index]] = kw_distances[0][index]

        keywords_dict = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)

        rdb.mset(data_insert_dict)
        
    except Exception as e:
        logging.error(e)
        logging.error(traceback.print_exc())

    return keywords_dict

def update_query_df(log_txt, query_df, file_path):

    logging.info(f"\n Finished {log_txt} .... ")
    query_df.to_pickle(file_path)

def get_cdd_label(page_id, df):
    
    label = df[df['page_id'] == page_id]['label'].values
    if len(label) == 0:
        return 5
    else:
        return label[0]

def read_document_data(filepath):

    try:
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
    except Exception as e:
        data_dict = dict()
        print(e)

    return data_dict

def get_cdd_df(query):

    query_updated = query.lower().replace(' ', '_')

    es_mr_data = read_document_data(DATA_PATH+f'search_results_index/{query_updated}_bm25_result.json')
    ss_mr_data = read_document_data(DATA_PATH+f'search_results_index/{query_updated}_semantic_result.json')

    cdd_page_id_list = list(set(list(es_mr_data.values()) + list(ss_mr_data.values())))
    cdd_df = xlm_df[xlm_df['page_id'].isin(cdd_page_id_list)]

    return cdd_df

def get_data_page_id(page_id, col_name):

    data_df = xlm_df[xlm_df['page_id'] == page_id]
    data = data_df[col_name].values[0]

    return data

def get_diff_sim(subtopic_keywords_list):

    sim_list = []
    for key, value in subtopic_keywords_list:
        sim_list.append(value)

    return round(max(sim_list)-min(sim_list), 3)

if __name__ == '__main__':

    logging.info("Started candidate pool generation ....")

    try:
        start = time.time()
        tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))
        cdd_type = 'large_cdd'

        if cdd_type == 'small_cdd':
            query_dataframe_folderpath_update = DATA_PATH+'dataframes/query_dataframes_updated_nc_small_cdd/'
        elif cdd_type == 'large_cdd':
            query_dataframe_folderpath_update = DATA_PATH+'dataframes/query_dataframes_updated_nc_large_cdd/'


        query_dataframe_folderpath = DATA_PATH+'dataframes/query_dataframes/'

        for query_df_filename in os.listdir(query_dataframe_folderpath):

            query_df = pd.read_pickle(query_dataframe_folderpath_update+query_df_filename)

            query = query_df['query'].values[0]
            # query_vec = tf_model(query)['outputs'].numpy()[0]

            logging.info(f'starting query: {query}')

            # if cdd_type == 'large_cdd':
                # cdd_df = get_cdd_df(query)
                # cdd_df['label'] = cdd_df.apply(lambda x:get_cdd_label(x['page_id'], query_df) , axis=1)
                # cdd_df['query'] = query_df['query'].values[0]
                # query_df = cdd_df
                # query_dataframe_folderpath_update = DATA_PATH+'dataframes/query_dataframes_updated_nc_large_cdd/'

            query_update_filepath = query_dataframe_folderpath_update+query_df_filename

            # if cdd_type == 'small_cdd':
            #     query_df['doc_repr_vec'] = query_df.apply(lambda x:get_data_page_id(x['page_id'], 'doc_repr_vec'), axis=1)
            #     update_query_df('doc_repr_vec', query_df, query_update_filepath)

            #     query_df['text_tokens'] = query_df.apply(lambda x:get_data_page_id(x['page_id'], 'text_tokens'), axis=1)
            #     update_query_df('extract_and_clean_ncs', query_df, query_update_filepath)

            #     query_df['keywords'] = query_df.apply(lambda x:get_data_page_id(x['page_id'], 'keywords'), axis=1)
            #     update_query_df('keywords', query_df, query_update_filepath)

            # query_df['text'] = query_df.apply(lambda x:get_clean_text(x['text']), axis=1)
            # query_df['doc_repr_vec'] = query_df.apply(lambda x:get_representation_vector(x['text'], x['title']), axis=1)
            # update_query_df('doc_repr_vec', query_df, query_update_filepath)

            # query_df['text_tokens'] = query_df.apply(lambda x:extract_and_clean_ncs(x['text'], language_flag=x['lang'], representation_level='TEXT'), axis=1)
            # update_query_df('extract_and_clean_ncs', query_df, query_update_filepath)

            # query_df['keywords'] = query_df.apply(lambda x:get_sent_transformers_keywords(x['doc_repr_vec'], x['text_tokens'], max_keyword_cnt = 25), axis=1)
            # update_query_df('keywords', query_df, query_update_filepath)

            # query_df['keywords_use'] = query_df.apply(lambda x:get_sent_transformers_keywords_use(x['keywords'], query_vec, max_keyword_cnt = 25), axis=1)
            # query_df['candidate_pool'] = query_df.apply(lambda x:get_candidate_pool(x['keywords_use'], lower_limit = 0.0, upper_limit = 0.45), axis=1)

            query_df['diff_sim'] = query_df.apply(lambda x:get_diff_sim(x['keywords_use']), axis=1)

            update_query_df('keywords_use and candidate_pool', query_df, query_update_filepath)

            logging.info(f'finished query: {query}')    

        end = time.time()
        logging.info(f"\n time taken .... {end-start} secs")

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

    logging.info("Finished candidate pool generation ....")