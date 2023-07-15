from settings_abstract import *

def sqlite_common_query_seq(sql_query, sql_select=False, sql_insert_params=None):

    print('running sqlite_common_query_seq')

    try:
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        if sql_insert_params:
            cursor.execute(sql_query, sql_insert_params)
        else:
            cursor.execute(sql_query)

        if sql_select:
            query_result = cursor.fetchall()
            conn.close()

            return query_result

        conn.commit()
        conn.close()

        print('finished sqlite_common_query_seq')
    except Exception as e:
        print(e)

def get_rank_from_rank_df(idx):
    # return rank_df[rank_df['id'] == idx]['gt_rank'].values[0]
    return None

def transform_gt_rank(rank):
    
    if rank == 1:
        return 3
    elif rank == 2:
        return 2
    elif rank == 3:
        return 0
    elif rank == 4:
        return 0

def get_gt_ranking(df):
    return df.gt_rank.values

def get_ndcg_scores(true_ranking, predicted_ranking):
    
    ndcg_15 = ndcg_score(true_ranking, predicted_ranking, k=15)
    ndcg_10 = ndcg_score(true_ranking, predicted_ranking, k=10)
    ndcg_5 = ndcg_score(true_ranking, predicted_ranking, k=5)
    
    print(f'NDCG@15 -- {ndcg_15}')
    print(f'NDCG@10 -- {ndcg_10}')
    print(f'NDCG@5 -- {ndcg_5}')

def query_docs(abstract):
    
    abstract = abstract.lower()
    if query_search_arxiv in abstract:
        return 1
    return 0

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_document_vec(text):
    
    return tf_model(text)['outputs'].numpy()[0].reshape(1, -1)

def get_pool_vec(doc_vec_list, pool):
    
    doc_vec_list = get_modified_vectors(doc_vec_list)
    if pool == 'mean':
        return np.mean(doc_vec_list, axis=0)
    elif pool == 'max':
        return np.amax(doc_vec_list, axis=0)
    
def get_cosine_sim(vec_1, vec_2):
    
    return cosine_similarity(vec_1.reshape(1, -1), vec_2.reshape(1, -1))[0][0]

def detect_language(text):
    
    lang_label = fasttext_model.predict(text.replace("\n"," "))[0][0].split('__label__')[1]
    return lang_label

def get_nounchunks_vector(nounchunk_list):
    
    nounchunk_vec_list = []
    for nc in nounchunk_list:
        nounchunk_vec_list.append(get_document_vec(nc))
        
    return get_pool_vec(np.array(nounchunk_vec_list), pool='mean')

def get_spacy_doc(text, lang):

    doc = None
    if len(text) >= 999999:
            return None

    if lang == 'en':
        doc = nlp_en(text)
    elif lang == 'de':
        doc = nlp_de(text)
    else:
        return None

    return doc

def get_noun_chunks(text, lang):
    
    try:
        doc = get_spacy_doc(text, lang)
        if doc is None:
            return None
        noun_phrases_list = []
        
        for nc in doc.noun_chunks:
            noun_phrases_list.append(nc.lemma_) 
            
        return noun_phrases_list
    except Exception as e:
        print(e)
        return None

def get_document_adjectives(text, lang):

    try:
        doc = get_spacy_doc(text, lang)
        if doc is None:
            return None
        adjs_list = []
        
        for token in doc:
            if token.pos_ == "ADJ":
                adjs_list.append(token.lemma_)

        return adjs_list
        
    except Exception as e:
        print(e)
        return None

def get_document_verbs(text, lang):

    try:
        doc = get_spacy_doc(text, lang)
        if doc is None:
            return None
        verbs_list = []
        
        for token in doc:
            if token.pos_ == "VERB":
                verbs_list.append(token.lemma_)

        return verbs_list
        
    except Exception as e:
        print(e)
        return None


def get_top_keywords(text):
    
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),
                              use_maxsum=True, nr_candidates=20, top_n=15)

def get_keywords_vector(keywords_list):
    
    keyword_vec_list = []
    for keyword in keywords_list:
        keyword_vec_list.append(get_document_vec(keyword[0]))
        
    return get_pool_vec(np.array(keyword_vec_list), pool='mean')

def get_summarization_text(text):
    return summarizer(text, min_length=50, max_length=100)[0]['summary_text']

def get_abstract_summarization_vectors(text):
    
    text_tokens = text.split()
    summarizer_vec_list = []
    
    if len(text_tokens) < 512:
        summarizer_vec_list.append(get_summarization_text(text))
    else:
        text_1 = ' '.join(text_tokens[:512])
        text_2 = ' '.join(text_tokens[512:])
        
        summarizer_vec_list.append(get_summarization_text(text_1))
        summarizer_vec_list.append(get_summarization_text(text_2))
        
    return get_nounchunks_vector(summarizer_vec_list)

def get_text_summarization_data(text):
    
    paragraph_vecs = []
    sentences = sent_tokenize(text)
    num_sent = 5
    
    for idx in range(0, num_sent, 5):
        idx_end = idx+5
        if idx_end < num_sent and num_sent-idx_end < 3:
            gt_sentences = sentences[idx:idx_end]
        else:
            gt_sentences = sentences[idx:]

        paragraph_vecs.append(get_document_vec(''.join(gt_sentences))) 
    
    sim_scores = []
    for paragraph_vec in paragraph_vecs:
        for gt_vec in arxiv_query_df.summarizer_vec:
            sim_scores.append(get_cosine_sim(paragraph_vec, gt_vec))

    return np.mean(np.array(sim_scores), axis=0)

def update_dataframes(idx):

    final_df.to_pickle(final_df_filepath)
    logging.info(f'final df loaded successfully ... {idx}')

    arxiv_query_df.to_pickle(arxiv_query_df_filepath)
    logging.info(f'arxiv query df loaded successfully ... {idx}')

def query_docs(abstract):
    
    if query_search_arxiv not in abstract.lower():
        return 'NA'
    
    sentences = sent_tokenize(abstract)
    sentences_list = []
    
    for sent in sentences:
        if query_search_arxiv in sent.lower():
            sentences_list.append(sent)
    
    if len(sentences_list) == 0:
        return 'NA'
    else:
        return ' '.join(sentences_list)

def get_articles():

    articles = []
    with open(arxiv_filepath, "r") as f:
        for l in f:
            d = json.loads(l)
            articles.append((d['id'], d['title'], d['abstract'], d['categories']))

    return articles

def query_docs_sent(abstract):
    
    if query_search_arxiv not in abstract.lower():
        return 'NA'
    
    sentences = sent_tokenize(abstract)
    sentences_list = []
    
    for sent in sentences:
        if query_search_arxiv in sent.lower():
            sentences_list.append(sent)
    
    if len(sentences_list) == 0:
        return 'NA'
    else:
        return ' '.join(sentences_list)

def query_docs_tf(abstract):
    
    return abstract.lower().count(query_search_arxiv)

if __name__ == '__main__':

    logging.info('starting arXiv abstract vector analysis ... ')

    try:

        query = 'robotik'
        query_search_arxiv = 'robot'
        query_updated = query.lower().replace(' ', '_')

        logging.info(f'\n\n Working on query {query} ... \n\n')

        select_table_query = """SELECT * FROM retrieval_dataset where query='"""+query+"""'"""

        with open(DATA_PATH+f'search_results_index/{query_updated}_bm25_result.json', 'r') as f:
            bm25_ranking = json.load(f)
            
        with open(DATA_PATH+f'search_results_index/{query_updated}_semantic_result.json', 'r') as f:
            semantic_ranking = json.load(f)

        dataframe_folder_path = DATA_PATH+f'dataframes/{query.lower()}'
        final_df_filepath = dataframe_folder_path+'/final_df.pkl'
        arxiv_query_df_filepath = dataframe_folder_path+'/arxiv_query_tf_df.pkl'
        # arxiv_query_df_sent_filepath = dataframe_folder_path+'/arxiv_query_sent_df.pkl'

        if not os.path.isdir(dataframe_folder_path):
            os.mkdir(dataframe_folder_path)

        final_df = pd.read_pickle(final_df_filepath)
        # arxiv_query_df = pd.read_pickle(arxiv_query_df_filepath)

        # query_result = sqlite_common_query_seq(select_table_query, sql_select=True) 
        # rank_df = pd.DataFrame(query_result, columns=['query', 'id', 'gt_rank'])

        # df_xlm = df_xlm[['id', 'text', 'lang', 'title', 'pubDate', 'label', 'text_len']]

        # final_df = pd.concat([rank_df.set_index('id'), df_xlm.set_index('id')], axis=1, join='inner').reset_index()
        # # final_df['gt_rank'] = final_df.apply(lambda x:get_rank_from_rank_df(x['id']), axis=1)
        # # final_df['gt_rank'] = final_df.apply(lambda x:transform_gt_rank(x['gt_rank']), axis=1)

        final_df.to_pickle(final_df_filepath)
        logging.info('final df loaded successfully ... 1')

        articles = get_articles()
        arxiv_df = pd.DataFrame(articles, columns=['id', 'title', 'abstract', 'categories'])

        # arxiv_df['abstract_sents'] = arxiv_df.apply(lambda x:query_docs_sent(x['abstract']), axis=1)
        # arxiv_df['abstract_len'] = arxiv_df.apply(lambda x:len(x['abstract_sents']), axis=1)
        # arxiv_query_df = arxiv_df[arxiv_df['abstract_len'] > 5]

        # arxiv_query_df = arxiv_df[arxiv_df['quant_label'] == 1]

        arxiv_df['quant_label'] = arxiv_df.apply(lambda x:query_docs_tf(x['abstract']), axis=1)
        arxiv_query_df = arxiv_df[arxiv_df['quant_label'] > 7]
        logging.info(f'arxiv_query_df index length ###### {len(arxiv_query_df.index)}')

        arxiv_query_df.to_pickle(arxiv_query_df_filepath)
        logging.info('arxiv query df loaded successfully ... 1')

        arxiv_query_df['doc_vec'] = arxiv_query_df.apply(lambda x:get_document_vec(x['abstract']), axis=1)
        # final_df['doc_vec'] = final_df.apply(lambda x:get_document_vec(x['text']), axis=1)

        logging.info('finished document vectors ..... ')
        update_dataframes(2)

        arxiv_query_df['lang'] = arxiv_query_df.apply(lambda x:detect_language(x['abstract']), axis=1)

        arxiv_query_df['noun_chunks'] = arxiv_query_df.apply(lambda x:get_noun_chunks(x['abstract'], x['lang']), axis=1)
        # final_df['noun_chunks'] = final_df.apply(lambda x:get_noun_chunks(x['text'], x['lang']), axis=1)

        arxiv_query_df['nounchunk_mean_vec'] = arxiv_query_df.apply(lambda x:get_nounchunks_vector(x['noun_chunks']), axis=1)
        # final_df['nounchunk_mean_vec'] = final_df.apply(lambda x:get_nounchunks_vector(x['noun_chunks']), axis=1)

        logging.info('finished noun chunk vectors ..... ')
        update_dataframes(3)

        arxiv_query_df['keywords'] = arxiv_query_df.apply(lambda x:get_top_keywords(x['abstract']), axis=1)
        # final_df['keywords'] = final_df.apply(lambda x:get_top_keywords(x['text']), axis=1)

        arxiv_query_df['keyword_mean_vec'] = arxiv_query_df.apply(lambda x:get_keywords_vector(x['keywords']), axis=1)
        # final_df['keyword_mean_vec'] = final_df.apply(lambda x:get_keywords_vector(x['keywords']), axis=1)

        logging.info('finished keyword vectors ..... ')
        update_dataframes(4)

        arxiv_query_df['summarizer_vec'] = arxiv_query_df.apply(lambda x:get_abstract_summarization_vectors(x['abstract']), axis=1) 
        arxiv_query_df.to_pickle(arxiv_query_df_filepath)
        logging.info('arxiv query df loaded successfully ... 5')

        final_df['mean_sim_summ'] = final_df.apply(lambda x:get_text_summarization_data(x['text']), axis=1)
        # final_df.to_pickle(final_df_filepath)
        logging.info('final df loaded successfully ... 5')

        arxiv_query_df['adjectives'] = arxiv_query_df.apply(lambda x:get_document_adjectives(x['abstract'], x['lang']), axis=1)
        # final_df['adjectives'] = final_df.apply(lambda x:get_document_adjectives(x['text'], x['lang']), axis=1)

        logging.info('finished adjectives ..... ')
        update_dataframes(6)

        arxiv_query_df['verbs'] = arxiv_query_df.apply(lambda x:get_document_verbs(x['abstract'], x['lang']), axis=1)
        # final_df['verbs'] = final_df.apply(lambda x:get_document_verbs(x['text'], x['lang']), axis=1)

        logging.info('finished verbs ..... ')
        update_dataframes(6)

    except Exception as e:
        logging.error(repr(e))
        logging.error(str(e))
    
    logging.info('finished arXiv abstract vector analysis ... ')