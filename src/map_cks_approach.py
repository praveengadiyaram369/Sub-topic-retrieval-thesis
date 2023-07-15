import os
import numpy as np
import pandas as pd
import logging
import pickle
import traceback

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

import hdbscan
import umap

DATA_PATH = '/usr/src/app/data/'
tf_model = hub.load(os.path.join(DATA_PATH, 'models/USE_model'))

query_dataframe_folderpath = DATA_PATH+'dataframes/query_dataframes_updated_nc_small_cdd/'
query_dataframe_cks_folderpath = DATA_PATH+'dataframes/cks_data/'

bm25_folderpath = DATA_PATH + 'retriever_output/bm25/'
semantic_pool_folderpath = DATA_PATH + 'retriever_output/semantic_pool/'
candidate_pool_folderpath = DATA_PATH + 'retriever_output/candidate_pool/'
optimized_candidate_pool_folderpath = DATA_PATH + 'retriever_output/optimized_candidate_pool/'

bm25_query_folderpath = query_dataframe_cks_folderpath + 'bm25/'
semantic_pool_query_folderpath = query_dataframe_cks_folderpath + 'semantic_pool/'
candidate_pool_query_folderpath = query_dataframe_cks_folderpath + 'candidate_pool/'
optimzed_candidate_pool_query_folderpath = query_dataframe_cks_folderpath + 'optimized_candidate_pool/'

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

def get_all_doc_ids():

    all_document_ids = []
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

        bm25_reranker_output = read_from_pickle(bm25_folderpath + query_updated +'.pickle')
        semantic_reranker_output = read_from_pickle(semantic_pool_folderpath + query_updated +'.pickle')
        candidate_pool_map_output = read_from_pickle(candidate_pool_folderpath + query_updated +'.pickle')

        all_document_ids.extend(bm25_reranker_output)
        all_document_ids.extend(semantic_reranker_output)
        all_document_ids.extend(candidate_pool_map_output)

    return list(set(all_document_ids))

def get_positive_doc_ids(df):

    doc_ids = []
    for idx, row in df.iterrows():

        label = row['label']
        doc_id = row['page_id']

        if label == 1:
            doc_ids.append(doc_id)

    return doc_ids

def calculate_map(query_ap_list):

    map_data = round(sum(query_ap_list)/len(query_ap_list), 3)
    return map_data


def get_candidate_pool(subtopic_keywords_list, cp_threshold = 0.4):
    
    sim_values = []
    for key, value in subtopic_keywords_list:
        sim_values.append(value)
            
    upper_limit = round(np.percentile(sim_values, cp_threshold), 3)
    candidate_pool = []

    for key, value in subtopic_keywords_list:
        
        if value < upper_limit:
            candidate_pool.append(key)
            
    return candidate_pool

def get_sent_transformers_keywords_use(keywords, query_vec, max_keyword_cnt = 30):
    
    keywords = list(dict(keywords).keys())
    
    # candidate_embeddings_keywords = [m.unpackb(rdb.get(kw)) for kw in keywords]
    candidate_embeddings_keywords = [tf_model(kw)['outputs'].numpy()[0] for kw in keywords]
        
    query_distances = cosine_similarity([query_vec], candidate_embeddings_keywords)
    subtopic_keywords_dict = dict()
    for index in query_distances.argsort()[0][-max_keyword_cnt:]: 
        
        subtopic_keywords_dict[keywords[index]] = query_distances[0][index]
    
    subtopic_keywords_dict = sorted(subtopic_keywords_dict.items(), key=lambda x: x[1], reverse=True)

    return subtopic_keywords_dict

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

def get_umap_output(vec_array, dim_size=5):
    
    umap_obj = umap.UMAP(n_neighbors=30, 
                        n_components=dim_size, 
                        min_dist=0.01,
                        metric='cosine',
                        random_state=123).fit(vec_array) 
    
    umap_output = umap_obj.transform(vec_array) 
    return umap_output, umap_obj

def get_hdbscan_output(data_points, min_cluster_size, min_samples, cluster_size=7):
    
    hdbscan_output = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric='euclidean',
                            cluster_selection_method='eom').fit(data_points)
    return hdbscan_output

def get_dbscan_output(data_points, cluster_size=7):
    
    dbscan_output = DBSCAN(
        #eps=3,
#                                       min_samples=2,
                                      metric='euclidean').fit(data_points)
    return dbscan_output

def project_on_2Dplane(umap_output, cluster_ids):
    
    umap_df = pd.DataFrame(np.column_stack((umap_output, cluster_ids)), columns=['x', 'y', 'cluster ids'])
    # grid = sns.FacetGrid(umap_df, hue='cluster ids', height=7)
    # grid.map(plt.scatter, 'x', 'y').add_legend()

def remove_noise(data_points, labels):
    
    data_points_filtered = []
    labels_filtered = []
    
    for dp, l in zip(data_points, labels):
        if l != -1:
            data_points_filtered.append(dp)
            labels_filtered.append(l)
            
    return data_points_filtered, labels_filtered

def get_clustering_analysis(cluster_df, final_candidate_pool_vecs, min_cluster_size, min_samples, dimen_size=5, cluster_size=7, clustering='hdbscan'):
    
    umap_output_5, umap_5 = get_umap_output(final_candidate_pool_vecs, dim_size=dimen_size)
    if clustering == 'hdbscan':
        clustering_output = get_hdbscan_output(umap_output_5, min_cluster_size, min_samples, cluster_size=cluster_size)
    elif clustering == 'dbscan':
        clustering_output = get_dbscan_output(umap_output_5, cluster_size=cluster_size)
    
    cluster_df['cluster_id'] = clustering_output.labels_
    data_points, labels = remove_noise(umap_output_5, clustering_output.labels_)
    sil_score = silhouette_score(np.array(data_points),np.array(labels),metric="euclidean",random_state=200)
    print(f'Silhouette score ... {sil_score}')

#     cluster_df.cluster_id.hist(bins=150)
    
    # umap_output_2, umap_2 = get_umap_output(final_candidate_pool_vecs, dim_size=2)
    # project_on_2Dplane(umap_output_2, cluster_df['cluster_id'])
    
    return cluster_df, sil_score

def get_nearest_keyword(keywords, keyword_vecs, mean_vec):
    
    query_distances = cosine_similarity([mean_vec], list(keyword_vecs))
    subtopic_keywords_dict = dict()
    for index in query_distances.argsort()[0]: 
        
        subtopic_keywords_dict[keywords[index]] = query_distances[0][index]
    
    subtopic_keywords_dict = sorted(subtopic_keywords_dict.items(), key=lambda x: x[1], reverse=True)
    return subtopic_keywords_dict[0][0]

def get_topic_documents(topic_words, final_df):
    
    doc_id_list = []
    for idx, row in final_df.iterrows():

        candidate_pool = row['candidate_pool']
        doc_id = row['id']

        for tw in topic_words:
            if tw in candidate_pool:
                doc_id_list.append(doc_id)

    return list(set(doc_id_list))

def get_meta_data(cluster_id, df, page_id_list):
    
    label_1 = label_2 = label_3 = label_4 = label_5 = 0
    
    for page_id in page_id_list:
        
        label = int(df[df['id'] == page_id]['label'].values[0])
        
        if label == 1:
            label_1 += 1
        elif label == 2:
            label_2 += 1
        elif label == 3:
            label_3 += 1
        elif label == 4:
            label_4 += 1
        elif label == 5:
            label_5 += 1

    return {
        'cluster_id': cluster_id,
        'label_1': label_1,
        'label_2': label_2,
        'label_3': label_3,
        'label_4': label_4,  
        'label_5': label_5,      
    }


def get_sub_topic_modelling(query_df, umap_dim, min_cluster_size, min_samples, clustering='hdbscan'):
    
    final_candidate_pool = []

    for idx, row in query_df.iterrows():
        final_candidate_pool.extend(row['candidate_pool'])
        
    final_candidate_pool = list(set(final_candidate_pool))

    # final_candidate_pool_vecs = [m.unpackb(rdb.get(nc)) for nc in final_candidate_pool]
    final_candidate_pool_vecs = [tf_model(nc)['outputs'].numpy()[0] for nc in final_candidate_pool]

    df_data = []
    for word, vec in zip(final_candidate_pool, final_candidate_pool_vecs):
        df_data.append((word, vec))

    cluster_df = pd.DataFrame(df_data, columns= ['candidate_words', 'candidate_vecs'])
    cluster_df, sil_score = get_clustering_analysis(cluster_df, final_candidate_pool_vecs, min_cluster_size, min_samples, dimen_size=umap_dim, cluster_size=8, clustering='hdbscan')

    cluster_data = []

    for cluster_id in set(cluster_df.cluster_id.values):

        if cluster_id != -1:
            df = cluster_df[cluster_df['cluster_id'] == cluster_id]
            cluster_data.append((cluster_id, df.candidate_words.values, df.candidate_vecs.values))

    cluster_data_df = pd.DataFrame(cluster_data, columns=['cluster_id', 'candidate_words', 'candidate_vecs'])
    cluster_data_df['mean_vec'] = cluster_data_df.apply(lambda x:get_pool_vec(x['candidate_vecs'], 'mean'), axis=1)
    cluster_data_df['topic'] = cluster_data_df.apply(lambda x:get_nearest_keyword(x['candidate_words'], x['candidate_vecs'], x['mean_vec']), axis=1)

    cluster_data_df['page_id_list'] = cluster_data_df.apply(lambda x:get_topic_documents(x['candidate_words'], query_df), axis=1)
    cluster_data_df['doc_cnt'] = cluster_data_df.apply(lambda x:len(x['page_id_list']), axis=1)


    final_page_id_list = []
    for page_id_list in cluster_data_df.page_id_list.values:
        final_page_id_list.extend(page_id_list)

    final_page_id_list = list(set(final_page_id_list))

    document_cnt_list = []
    for page_id in final_page_id_list:

        page_id_cnt = 0
        for page_id_list in cluster_data_df.page_id_list.values:
            if page_id in page_id_list:
                page_id_cnt += 1

        document_cnt_list.append(page_id_cnt)


    meta_data = []
    for idx, row in cluster_data_df.iterrows():
        meta_data.append(get_meta_data(row['cluster_id'], query_df, row['page_id_list']))
        
    entity_df = pd.DataFrame(meta_data)
    cluster_data_df = pd.concat([cluster_data_df.set_index('cluster_id'), entity_df.set_index('cluster_id')], axis=1, join='inner').reset_index()
      
    return cluster_data_df, sil_score

def get_cluster_ranked_documents(cluster_df, query_df):

    document_list = []
    for idx, row in cluster_df.iterrows():

        page_id_list = row['page_id_list']
        page_id_df = query_df[query_df['id'].isin(page_id_list)]
        final_page_id_ranked = list(page_id_df.id.values)

        for page_id in final_page_id_ranked:
            if page_id not in document_list:
                document_list.append(page_id)

    return document_list

def get_expectation_average(df, target_ranks_list, query, system_name):

    query_avg_exp = {}
    pos_doc_cnt = 0

    query_updated = query.lower().replace(' ', '_')
    avg_prec_folder = DATA_PATH +f'average_precision_query/{system_name}/{query_updated}_avg_prec.json'
    
    for idx, row in df.iterrows():
        if row['label'] in target_ranks_list:
            pos_doc_cnt += 1
            query_avg_exp[idx+1] = round((pos_doc_cnt/(idx+1)), 3)

    avg_exp_val = list(query_avg_exp.values())
    if len(avg_exp_val) == 0:
        avg_exp = 0
    else:
        avg_exp = sum(avg_exp_val)/len(avg_exp_val)
    
    return avg_exp

def get_expectation_score_ordered_df(df, target_ranks_list, query, system_name):

    exp_score_5 = 0
    exp_score_10 = 0
    exp_score_15 = 0
    exp_score_20 = 0
    exp_score_25 = 0
    
    for idx, row in df.iterrows():
        
        if row['label'] in target_ranks_list:
            if idx < 5:
                exp_score_5 += 1
            elif idx < 10:
                exp_score_10 += 1
            elif idx < 15:
                exp_score_15 += 1
            elif idx < 20:
                exp_score_20 += 1
            elif idx < 25:
                exp_score_25 += 1
    
    exp_score_10 += exp_score_5
    exp_score_15 += exp_score_10
    exp_score_20 += exp_score_15
    exp_score_25 += exp_score_20

    exp_score_avg = get_expectation_average(df, target_ranks_list, query, system_name)
    
    # exp_score_5 /= 5
    # exp_score_10 /= 10
    # exp_score_15 /= 15
    # exp_score_20 /= 20
    # exp_score_25 /= 25

    exp_data =  {
        'exp_5': round(exp_score_5, 3),
        'exp_10': round(exp_score_10, 3),
        'exp_15': round(exp_score_15, 3),       
        'exp_20': round(exp_score_20, 3),
        'exp_25': round(exp_score_25, 3),
        'exp_25': round(exp_score_25, 3),
        'exp_avg': round(exp_score_avg, 3),
    }

    return exp_data

def get_ranked_df(df, query_vec, col_name):

    rank_df = df.copy()
    if col_name == 'mean_vec':
        rank_df['query_sim'] = rank_df.apply(lambda x:cosine_similarity(query_vec, get_modified_vectors(x[col_name]))[0][0], axis=1)
    elif col_name == 'nc_vec':
        rank_df['query_sim'] = rank_df.apply(lambda x:cosine_similarity(query_vec, x[col_name])[0][0], axis=1)
    elif col_name == 'doc_cnt':
        rank_df['query_sim'] = rank_df.apply(lambda x:len(x['page_id_list']), axis=1)

    rank_df = rank_df.sort_values(by=['query_sim'], ascending=False)
    rank_df = rank_df.reset_index(drop=True)

    return rank_df

def get_ordered_df(cluster_df, df, rank_type, cluster_rank_type, user_query, system_name):

    if cluster_rank_type == 'query':
        query = user_query
    elif cluster_rank_type == 'template':
        query = "Innovation and Technology"

    user_query = user_query.lower().replace(' ', '_')

    query_vec = get_modified_vectors(tf_model(query)['outputs'].numpy()[0])

    if rank_type == 'query_sim':
        ir2_df = get_ranked_df(cluster_df, query_vec, 'mean_vec')
    elif rank_type == 'doc_cnt':
        ir2_df = get_ranked_df(cluster_df, query_vec, rank_type)
    
    return ir2_df, query_vec, query


def get_cluster_ranking_expectation_scores(cluster_df, df, rank_type, cluster_rank_type, target_ranks_list, user_query, system_name):

    ir2_df, query_vec, query = get_ordered_df(cluster_df, df, rank_type, cluster_rank_type, user_query, system_name)    

    query_rank_df = get_ranked_df(df, query_vec, 'nc_vec')

    ranking_system_filepath = DATA_PATH +f'dataframes/ranking_data/{user_query}_{system_name}_ranking.pkl'
    ir2_df.to_pickle(ranking_system_filepath)

    ranked_document_list = get_cluster_ranked_documents(ir2_df, query_rank_df)

    df_list = []
    for page_id in ranked_document_list:
        df_list.append(df[df['id']==page_id])

    ordered_df = pd.concat(df_list)
    ordered_df = ordered_df.reset_index(drop=True)

    return get_expectation_score_ordered_df(ordered_df, target_ranks_list, query, system_name)

def get_elimination_cluster_cnt(cluster_df, df, sil_score, query, target_ranks_list=[1], targeted_rank='1'):
    
    # cluster_cnt_eliminated = 0
    # neg_doc_eliminated = []
    # # query = df['query'].values[0]
    
    # negative_page_id_list = []
    # for page_id, label in zip(df.page_id.values, df.label.values):
    #     if label == 3 or label == 4:
    #         negative_page_id_list.append(page_id)
        
    # total_doc_cnt = len(set(negative_page_id_list))
    
    # for idx, row in cluster_df.iterrows():
        
    #     label_1 = row['label_1']
    #     label_2 = row['label_2']
    #     label_3 = row['label_3']
    #     label_4 = row['label_4']
    #     label_5 = row['label_5']

    #     criteria = (2*label_2) - (label_3+label_4)
    #     if label_1 == 0 and criteria < 0:
            
    #         cluster_cnt_eliminated += 1
    #         common_neg_docs = list(set(negative_page_id_list) & set(row['page_id_list']))
    #         neg_doc_eliminated.extend(common_neg_docs)
    
    # targeted_doc_cnt = len(set(neg_doc_eliminated))
    
    # clustered_page_id_list = []
    # for id_list in cluster_df.page_id_list.values:
    #     clustered_page_id_list.extend(id_list)
    
    # cluster_page_ids = set(clustered_page_id_list)
    # all_page_ids = set(df.page_id.values)
    
    # clustering_loss = all_page_ids - cluster_page_ids
    # missed_documents = get_document_labels(clustering_loss, df)

    # ir0_expectation_data_0 = get_query_expectation_scores_random(df, target_ranks_list)
    # ir1_expectation_data_1 = get_query_expectation_scores(df, target_ranks_list, system_name='IR1')
    # ir2_expectation_data_2 = get_cluster_ranking_expectation_scores(cluster_df, df, 'query_sim', 'query', target_ranks_list, query, system_name='IR2')
    # ir2_expectation_data_3 = get_cluster_ranking_expectation_scores(cluster_df, df, 'query_sim', 'template', target_ranks_list, query, system_name='IR3')
    ir2_expectation_data_4 = get_cluster_ranking_expectation_scores(cluster_df, df, 'doc_cnt', 'query', target_ranks_list, query, system_name='IR4')

    # ir2_expectation_data_5 = get_cluster_expectation_scores_combined_two(cluster_df, df, 'query', target_ranks_list, query, system_name='IR6')
    # ir2_expectation_data_6 = get_cluster_expectation_scores_combined_three(cluster_df, df, 'query', target_ranks_list, query, system_name='IR7')

    # ir2_expectation_data_7 = get_cluster_expectation_scores(cluster_df, df, target_ranks_list, system_name='IR5')

    return ir2_expectation_data_4

def calculate_ap(filename, df, query_doc_ids, query_vec, query_updated, retriever_type, query):

    retriever_data = read_from_pickle(filename)
    list_slice_idx = [passage_id_list[idx] for idx in retriever_data]

    if retriever_type == 'bm25':
        query_df_path = bm25_query_folderpath+query_updated+'.pkl'
    elif retriever_type == 'semantic_pool':
        query_df_path = semantic_pool_query_folderpath+query_updated+'.pkl'
    elif retriever_type == 'candidate_pool':
        query_df_path = candidate_pool_query_folderpath+query_updated+'.pkl'

    if os.path.isfile(query_df_path):
        query_df = pd.read_pickle(query_df_path)
    else:
        query_df = df[df['id'].isin(list_slice_idx)]
        query_df['label'] = query_df.apply(lambda x:1 if x['id'] in query_doc_ids else 0, axis=1)

        query_df['keywords_use'] = query_df.apply(lambda x:get_sent_transformers_keywords_use(x['keywords'], query_vec, max_keyword_cnt = 25), axis=1)
        query_df['candidate_pool'] = query_df.apply(lambda x:get_candidate_pool(x['keywords_use'], cp_threshold = cp_threshold), axis=1)
        
        query_df.to_pickle(query_df_path)

    cluster_data_df, sil_score = get_sub_topic_modelling(query_df, umap_dim, min_cluster_size, min_samples, clustering='hdbscan')
    clustering_output = get_elimination_cluster_cnt(cluster_data_df, query_df, sil_score, query, target_ranks_list=[1], targeted_rank='1')

    return clustering_output['exp_avg']

if __name__ == '__main__':

    logging.info("Starting retriever data generator ... ")

    try:

        xlm_df = pd.read_pickle(DATA_PATH+'final_keywords_dataframe_cdd.pkl')
        all_document_ids = get_all_doc_ids()

        cp_threshold = 65
        umap_dim = 5
        min_cluster_size = 20
        min_samples = 10

        passage_id_list = []
        for idx, row in xlm_df.iterrows():

            passage_id = row['id']
            passage_id_list.append(passage_id)

        logging.info(f"Passages: {len(passage_id_list)}")

        list_slice_idx = [passage_id_list[idx] for idx in all_document_ids]

        xlm_df = xlm_df[xlm_df['id'].isin(list_slice_idx)]

        logging.info(f'xlm_df length: {len(xlm_df.index)}')

        target_queries = ['Kryptologie', 'Defense', 'Cyber Attack', 'Data Centric Warfare', 'unbemannte Wirksysteme', ]
        bm25_map_list = []
        semantic_pool_map_list = []
        candidate_pool_map_list = []
        optimized_candidate_pool_map_list = []

        for query_df_filename in os.listdir(query_dataframe_folderpath):

            query_df = pd.read_pickle(query_dataframe_folderpath+query_df_filename)
            query = query_df['query'].values[0]
            query_vec = tf_model(query)['outputs'].numpy()[0]

            if query in target_queries:
                continue

            query_updated = query.lower().replace(' ', '_')
            query_doc_ids = get_positive_doc_ids(query_df)

            # bm25_map_output = calculate_ap(bm25_folderpath + query_updated +'.pickle', xlm_df, query_doc_ids, query_vec, query_updated, 'bm25', query)

            # semantic_map_output = calculate_ap(semantic_pool_folderpath + query_updated +'.pickle', xlm_df, query_doc_ids, query_vec, query_updated, 'semantic_pool', query)

            # candidate_pool_map_output = calculate_ap(candidate_pool_folderpath + query_updated +'.pickle', xlm_df, query_doc_ids, query_vec, query_updated, 'candidate_pool', query)

            optimzed_candidate_pool_map_output = calculate_ap(candidate_pool_folderpath + query_updated +'.pickle', xlm_df, query_doc_ids, query_vec, query_updated, 'candidate_pool', query)

            # bm25_map_list.append(bm25_map_output)
            # semantic_pool_map_list.append(semantic_map_output)
            # candidate_pool_map_list.append(candidate_pool_map_output)
            optimized_candidate_pool_map_list.append(optimzed_candidate_pool_map_output)

            logging.info(f"Finished {query} ... ")

        # bm25_map = calculate_map(bm25_map_list)
        # semantic_pool_map = calculate_map(semantic_pool_map_list)
        # candidate_pool_map = calculate_map(candidate_pool_map_list)
        optimized_candidate_pool_map = calculate_map(optimized_candidate_pool_map_list)


        # map_output_data = [bm25_map, semantic_pool_map, candidate_pool_map]
        map_output_data = [optimized_candidate_pool_map]
        write_to_pickle(DATA_PATH+'map_output_data_cks.pickle', map_output_data)

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())

    logging.info("Finished retriever data generator ... ")