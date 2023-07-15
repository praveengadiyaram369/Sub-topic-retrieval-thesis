import os
import json
import pickle
import logging

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# DATA_PATH = os.getcwd() + '/../data/'
DATA_PATH = '/usr/src/app/data/'

LOG_FILE = DATA_PATH + f'logs/feature_extraction_tracker_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

multi_model = pickle.load(open(DATA_PATH+'/models/multi_model_stage_1.pkl', 'rb'))
bin_model = pickle.load(open(DATA_PATH+'/models/bin_model_stage_2.pkl', 'rb'))
threshold = 0.109

nc_list_filepath = DATA_PATH + f'nc_features/'
pickle_filepath = DATA_PATH + f'unlabeled_data_feature_extracted.pkl'


def write_data_to_file(filepath, data):

    with open(filepath, "a") as f:
        f.write(data+'\n')

def write_document_data(data, filepath):

    with open(filepath, 'w') as f:
        json.dump(data, f)

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_threshold_output(preds, threshold):
    
    y_preds = []
    for val in preds:
        if val >= threshold:
            y_preds.append(1)
        else:
            y_preds.append(0)
        
    y_preds = np.array(y_preds)
    
    return y_preds

def filter_unlabeled_data():
    
    try:
        X_unlabeled = get_modified_vectors(unlabeled_df.nc_vec.values)
        unlabeled_df['milt_label'] = multi_model.predict(X_unlabeled)

        unlabeled_df_new = unlabeled_df[unlabeled_df['milt_label'].isin([0,1,3])]
        X_unlabeled_new = get_modified_vectors(unlabeled_df_new.nc_vec.values)

        preds = bin_model.predict_proba(X_unlabeled_new)[:,1]
        unlabeled_df_new['tech_label'] = get_threshold_output(preds, threshold)

        tech_df = unlabeled_df_new[unlabeled_df_new['tech_label'] == 1]
        milt_df = unlabeled_df[unlabeled_df['milt_label'] == 2]

        tech_df.to_pickle(DATA_PATH+'tech_df_final.pkl')
        milt_df.to_pickle(DATA_PATH+'milt_df_final.pkl')
    except Exception as e:
        logging.error(e)

def load_txt_files(folder_path):

    file_data = []
    for file_name in os.listdir(folder_path):
        
        nc_vec = np.loadtxt(os.path.join(folder_path, file_name), delimiter=',').reshape(1, -1)
        file_data.append((file_name, nc_vec))

    return file_data

if __name__ == '__main__':

    logging.info('starting classification on unlabeled data ... ')

    try:
        all_files_data = load_txt_files(nc_list_filepath)
        vec_df = pd.DataFrame(all_files_data, columns=['id', 'nc_vec'])
        vec_df.to_pickle(DATA_PATH+'noun_chunk_vector_df.pkl')

        logging.info('finished noun chunk dataframe ... ')

        unlabeled_df = pd.read_pickle(pickle_filepath)

        unlabeled_df = unlabeled_df.merge(vec_df, how='left', on='id')
        unlabeled_df = unlabeled_df.dropna()
        unlabeled_df.to_pickle(DATA_PATH+'noun_chunk_vector_df.pkl')
        logging.info('finished merging dataframe ... ')

        filter_unlabeled_data()
    except Exception as e:
        logging.error(e)

    logging.info('finished classification on unlabeled data ... ')
