import os
import logging
import concurrent.futures
import time

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

import spacy
import fasttext
import fasttext.util

import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = os.getcwd() + '/../data/'
DATA_PATH = '/usr/src/app/data/'

tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
LOG_FILE = DATA_PATH + f'logs/feature_extraction_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

unlabeled_doc_filepath = DATA_PATH + f'cleaned_html_pages'
pickle_filepath = DATA_PATH + f'unlabeled_data_feature_extracted.pkl'
nc_list_filepath = DATA_PATH + f'nc_features/'
PRETRAINED_FASTTRACK_MODEL = os.path.join(DATA_PATH, 'models/lid.176.bin')

LANG_EN = "__label__en"
LANG_DE = "__label__de"

fasttext.FastText.eprint = lambda x: None
fasttext_model = fasttext.load_model(PRETRAINED_FASTTRACK_MODEL)

def detect_language(text):
    
    lang_label = fasttext_model.predict(text)[0][0].split('__label__')[1]
    return lang_label

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_avg_token_vector(token_list):

    page_id, tokens = token_list

    try:

        if tokens is not None:
            avg_token_vec = []
            for token in tokens:
                avg_token_vec.append(tf_model(token)['outputs'].numpy()[0].reshape(1, -1))
            
            final_feature_vec = get_modified_vectors(np.mean(avg_token_vec, axis=0))
            np.savetxt(nc_list_filepath+page_id, final_feature_vec, delimiter=',')


    except Exception as e:
        logging.error(e)
        return None

    return None


def load_txt_files(folder_path):

    file_data = []
    for idx, file_name in enumerate(os.listdir(folder_path)):
        
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors="ignore") as f:
            file_data.append((file_name, f.read()))

    return file_data

def get_document_data(text, lang):
    
    doc = None

    try:
        if len(text) >= 999999:
            return None

        if lang == 'en':
            doc = nlp_en(text)
        elif lang == 'de':
            doc = nlp_de(text)
        else:
            return None
        
        noun_phrases_list = []
        
        for nc in doc.noun_chunks:
            noun_phrases_list.append(nc.lemma_) 
    except Exception as e:
        logging.error(e)
        return None

            
    return noun_phrases_list

def load_processed_doc_list():

    doc_name_list = []
    for filename in os.listdir(nc_list_filepath):
        doc_name_list.append(filename)

    return doc_name_list


if  __name__ == '__main__':

    logging.info('starting feature extractor on unlabeled data ... ')
    start = time.perf_counter()

    try:
        
        processed_doc_list = load_processed_doc_list()
        # unlabeled_data = load_txt_files(unlabeled_doc_filepath)
        # unlabeled_df = pd.DataFrame(unlabeled_data, columns= ['id', 'text'])

        # unlabeled_df.to_pickle(pickle_filepath)   
        # logging.info('finished loading the dataframe ... ')

        # unlabeled_df['text_len'] = unlabeled_df.apply(lambda x:len(x['text'].split()), axis=1)
        # unlabeled_df = unlabeled_df[unlabeled_df['text_len'] > 40]

        # unlabeled_df.to_pickle(pickle_filepath)
        # logging.info('finished filtering short documents ... ')

        # unlabeled_df['lang'] = unlabeled_df.apply(lambda x:detect_language(x['text'].replace("\n"," ")), axis=1)
        # print(unlabeled_df.lang.value_counts())
        # unlabeled_df.to_pickle(pickle_filepath)
        # logging.info('finished language detection ... ')

        nlp_de = spacy.load("de_core_news_sm")
        nlp_en = spacy.load("en_core_web_sm")

        unlabeled_df = pd.read_pickle(pickle_filepath)

        # unlabeled_df['text_tokens'] = unlabeled_df.apply(lambda x:get_document_data(x['text'], x['lang']), axis=1)
        # # unlabeled_df.dropna(inplace=True)
        # unlabeled_df.to_pickle(pickle_filepath)
        # logging.info('finished noun-chunk extraction ... ')
        
        # unlabeled_df['nc_vec'] = unlabeled_df.apply(lambda x:get_avg_token_vector((x['id'], x['text_tokens'])), axis=1)

        final_list = []
        noun_chunk_list = unlabeled_df.text_tokens.values
        id_list = unlabeled_df.id.values

        for id_data, nc_data in zip(id_list, noun_chunk_list):
            if id_data not in processed_doc_list:
                final_list.append((id_data, nc_data))

                # get_avg_token_vector((id_data, nc_data))

        logging.info(f'Processed document list length: {len(processed_doc_list)}')
        logging.info(f'Unlabeled document list length: {len(id_list)}')
        logging.info(f'Remaining document list length: {len(final_list)}')

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            try:
                executor.map(get_avg_token_vector, final_list)
            except Exception as e:
                logging.error(e)



        logging.info('finished noun chunk feature extraction ... ')

    except Exception as e:
        logging.error(e)

    finish = time.perf_counter()
    logging.info(f'Finished running in {round(finish-start, 2)} sec')

    logging.info('Ending feature extractor on unlabeled data ... ')