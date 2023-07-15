import os
import string
import json
import numpy as np
import pickle
import traceback

import msgpack
import msgpack_numpy as m
m.patch()

import redis
import pandas as pd
from nltk.corpus import stopwords

import re
import cleantext

from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

import logging
import concurrent.futures

DOC_COUNTER = 0

DATA_PATH = '/usr/src/app/data/'
pickle_filepath = DATA_PATH+'final_dataframe.pkl'
pickle_write_filepath = DATA_PATH+'final_keywords_dataframe.pkl'
keyword_list_filepath = DATA_PATH + f'keyword_features_pkl/'


tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))

rdb = redis.StrictRedis(
    host='redis_cache',
    port=6379,
)

LOG_FILE = DATA_PATH + f'logs/redisdb_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)


def get_all_stopwords():

    stopwords_de = stopwords.words('german')
    stopwords_en = stopwords.words('english')

    stopwords_full = []
    stopwords_full.extend(stopwords_de)
    stopwords_full.extend(stopwords_en)

    stopwords_full = [word.lower() for word in stopwords_full]

    stop_all = set(stopwords_full + list(string.punctuation))

    return stop_all

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

def get_representation_vector(document, title):

    global DOC_COUNTER
    DOC_COUNTER += 1
    
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

    logging.info(f'finished document {DOC_COUNTER} ... ')
        
    return get_pool_vec(get_modified_vectors(doc_vecs), pool='mean')

def get_shorter_text(phrase_1, phrase_2):
    
    if len(phrase_1) < len(phrase_2):
        return phrase_1
    else:
        return phrase_2
    
def remove_stopwords(noun_chunks):
    
    filtered_noun_chunks = []
    stop_all = get_all_stopwords()
    for word_token in noun_chunks:
        if word_token.lower() not in stop_all:
            filtered_noun_chunks.append(word_token)
            
    return filtered_noun_chunks

def get_filtered_nc(noun_chunks):
    
    noun_chunks = list(set(noun_chunks))
    noun_chunks = remove_stopwords(noun_chunks)
    phrases_len = len(noun_chunks)
    remove_phrases = [] 

    for idx_1 in range(phrases_len):

        phrase_1 = noun_chunks[idx_1]
        for idx_2 in range(idx_1 + 1, phrases_len):
            phrase_2 = noun_chunks[idx_2]

            if fuzz.ratio(phrase_1, phrase_2) > 85:
                remove_phrases.append(get_shorter_text(phrase_1, phrase_2))

    final_noun_chunks = list(set(noun_chunks) - set(remove_phrases))
    return final_noun_chunks

def get_sent_transformers_keywords(page_id, repr_vec, noun_chunks, max_keyword_cnt = 30):

    global DOC_COUNTER
    DOC_COUNTER += 1

    keywords_dict = dict()
    
    try:
        noun_chunks = get_filtered_nc(noun_chunks)
        candidate_embeddings = [tf_model(nc)['outputs'].numpy()[0] for nc in noun_chunks]
        
        kw_distances = cosine_similarity([repr_vec], candidate_embeddings)
        
        data_insert_dict = dict()
        keywords_dict = dict()
        for index in kw_distances.argsort()[0][-max_keyword_cnt:]: 
            
            data_insert_dict[noun_chunks[index]] =  m.packb(candidate_embeddings[index])
                    
            keywords_dict[noun_chunks[index]] = kw_distances[0][index]

        with open(keyword_list_filepath+page_id+'.pkl', 'wb') as f:
            pickle.dump(keywords_dict, f)

        keywords_dict = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)

        rdb.mset(data_insert_dict)
        logging.info(f"Finished inserting into redis db ....{DOC_COUNTER}")
        
    except Exception as e:
        logging.error(e)
        logging.error(traceback.print_exc())

    return keywords_dict

def get_keywords_saveto_redis(final_list):

    page_id, noun_chunks, repr_vec = final_list
    max_keyword_cnt = 30
    
    noun_chunks = get_filtered_nc(noun_chunks)
    candidate_embeddings = [tf_model(nc)['outputs'].numpy()[0] for nc in noun_chunks]
    
    kw_distances = cosine_similarity([repr_vec], candidate_embeddings)
    
    data_insert_dict = dict()
    keywords_dict = dict()
    for index in kw_distances.argsort()[0][-max_keyword_cnt:]: 
        
        data_insert_dict[noun_chunks[index]] =  m.packb(candidate_embeddings[index])
                
        keywords_dict[noun_chunks[index]] = kw_distances[0][index]
        
    rdb.mset(data_insert_dict)
    logging.info(f"Finished inserting into redis db ....")

    with open(keyword_list_filepath+page_id+'.pkl', 'wb') as f:
        pickle.dump(keywords_dict, f)

    logging.info(f"Finished writing json ....")


def get_clean_text(text):
    
    clean_text = cleantext.clean(text,
            clean_all= False, # Execute all cleaning operations
            extra_spaces=True ,  # Remove extra white spaces 
            )
    
    clean_text = re.sub(r'http\S+', '', clean_text)
    p = re.compile(r'<.*?>')
    return p.sub('', clean_text)

if __name__ == '__main__':

    logging.info("Started redis db data insertion ....")

    try:
        df_xlm = pd.read_pickle(pickle_write_filepath)

        df_xlm['text'] = df_xlm.apply(lambda x:get_clean_text(x['text']), axis=1)
        stop_all = get_all_stopwords()

        df_xlm['doc_repr_vec'] = df_xlm.apply(lambda x:get_representation_vector(x['text'], x['title']), axis=1)
        df_xlm.to_pickle(pickle_write_filepath)

        logging.info("Document representation vector finished ....")

        final_list = []
        id_list = df_xlm.id.values
        noun_chunk_list = df_xlm.text_tokens.values
        doc_vec_list = df_xlm.doc_repr_vec.values

        for id_data, nc_data, doc_vec in zip(id_list, noun_chunk_list, doc_vec_list):
            final_list.append((id_data, nc_data, doc_vec))

        logging.info("Before ThreadPoolExecutor ....")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            try:
                executor.map(get_keywords_saveto_redis, final_list)
            except Exception as e:
                logging.error(e)
        
        logging.info("After ThreadPoolExecutor ....")

        df_xlm['keywords'] = df_xlm.apply(lambda x:get_sent_transformers_keywords(x['id'], x['doc_repr_vec'], x['text_tokens'], max_keyword_cnt = 30), axis=1)

        df_xlm.to_pickle(pickle_write_filepath)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.print_exc())

    logging.info("Finished redis db data insertion ....")