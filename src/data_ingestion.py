import os
import json
import time
import logging
import numpy as np
import pandas as pd

import faiss
from elasticsearch import helpers, Elasticsearch

import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = os.getcwd() + '/../data/'
DATA_PATH = '/usr/src/app/data/'
LOG_FILE = DATA_PATH + f'logs/data_ingestion_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

index = 'xxx_scraped_docs'
pipeline = 'multilang_pipe'

def get_label_name(label):
    
    if label == 1:
        return 'technology'
    elif label == 2:
        return 'military'
    
def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def doc_actions(data):

    for row in data:
        yield {
                '_index': index,
                'pipeline': pipeline,
                '_source': row,
                '_id': row['id']
            }

def ingest_to_elasticsearch():

    username = 'elastic'
    password = 'mit22yyy!'

    hostname = 'elasticsearch'
    port = '9200'

    try:
        time.sleep(10)
        es = Elasticsearch([f"http://{username}:{password}@{hostname}:{port}"], timeout=300, max_retries=10, retry_on_timeout=True)

        with open(DATA_PATH+'config/lpf_index_mappings.json', 'r') as f:
            index_mapping_lpf = json.load(f)
            
        with open(DATA_PATH+'config/lpf_pipeline.json', 'r') as f:
            pipeline_lpf = json.load(f)

        es.indices.delete(index=index, ignore=404)
        es.indices.create(index=index, body=index_mapping_lpf)

        es.ingest.delete_pipeline(id=pipeline, ignore=404)
        es.ingest.put_pipeline(id=pipeline, body=pipeline_lpf)

        df_es = df_xlm[['id', 'text', 'label_name', 'title', 'pubDate', 'url']]
        df_es = df_es.rename(columns={'text': 'contents', 'label_name':'label', 'pubDate':'published_date', 'url':'page_url'})
        doc_dict = df_es.to_dict(orient='records')

        logging.info('before running helpers.bulk')
        
        helpers.bulk(es, doc_actions(doc_dict), chunk_size=500, request_timeout=600, refresh='wait_for')

        logging.info('after running helpers.bulk')

    except Exception as e:
        logging.error(e)

def ingest_to_faiss(df_lang, lang):

    try:
        doc_embeddings = get_modified_vectors(df_lang.nc_vec.values)
        doc_embeddings = np.float32(doc_embeddings)

        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(doc_embeddings)

        faiss.write_index(index,DATA_PATH + f'{lang}_vector.index')

    except Exception as e:
        logging.error(e)

if __name__ == '__main__':

    logging.info('starting data ingestion ... ')

    try:
        tech_df = pd.read_pickle(DATA_PATH + 'tech_df_final.pkl')
        tech_df = tech_df.drop('milt_label', axis=1)

        milt_df = pd.read_pickle(DATA_PATH + 'milt_df_final.pkl')

        rssitem_df = pd.read_pickle(DATA_PATH +'rss_cache_df.pkl')

        tech_df = tech_df.rename(columns={'tech_label': 'label'})
        milt_df = milt_df.rename(columns={'milt_label': 'label'})

        df_xlm = pd.concat([tech_df, milt_df])
        df_xlm['label_name'] = df_xlm.apply(lambda x:get_label_name(x['label']), axis=1)
        df_xlm['id'] = df_xlm.apply(lambda x:x['id'][:-4], axis=1)

        df_xlm = df_xlm.merge(rssitem_df, how='left', on='id')
        df_xlm['pubDate'] = pd.to_datetime(df_xlm['pubDate'],unit='s')

        df_xlm.to_pickle(DATA_PATH+'final_dataframe.pkl')
        # df_xlm = pd.read_pickle(DATA_PATH+'final_dataframe.pkl')

        logging.info('finished loading dataframe ... ')

        ingest_to_elasticsearch()
        logging.info('finished loading into elastic search ... ')

        df_en = df_xlm[df_xlm['lang'] == 'en']
        df_de = df_xlm[df_xlm['lang'] == 'de']

        df_xlm.to_pickle(DATA_PATH+'xlm_dataframe.pkl')
        df_en.to_pickle(DATA_PATH+'en_dataframe.pkl')
        df_de.to_pickle(DATA_PATH+'de_dataframe.pkl')

        ingest_to_faiss(df_en, 'en')
        ingest_to_faiss(df_de, 'de')
        ingest_to_faiss(df_xlm, 'xlm')
        logging.info('finished loading into faiss ... ')

    except Exception as e:
        logging.error(e)

    logging.info('finished data ingestion ... ')