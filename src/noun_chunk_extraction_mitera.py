import pickle
import logging
import spacy
import traceback

import string
import re

from fuzzywuzzy import fuzz

DATA_PATH = '/usr/src/app/data/'

nlp_de = spacy.load("de_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

with open (DATA_PATH + 'pickle_files/stopwords_xxx.pkl', 'rb') as fp:
    all_stopwords = pickle.load(fp)

def portion_of_capital_letters(w):  # method belogs to abbreviation extraction
    upper_cases = ''.join([c for c in w if c.isupper()])
    return len(upper_cases) / len(w)


def check_if_word_is_abbreviation(word):  # method belogs to abbreviation extraction
    return (len(word.split()) == 1 and len(word) <= 13 and portion_of_capital_letters(word) >= 0.29)

def clean_string_from_puntuation_and_special_chars(s):
    invalidcharacters = set(string.punctuation)
    if any(char in invalidcharacters for char in s):
        s_ = s.translate(str.maketrans('', '', string.punctuation))
    else:
        s_ = s
    return s_


def remove_determiners(nc, language_flag):

    try:
        nc_splitted = nc.split()
        if language_flag == "EN":
            irrelevant_determiners = ["each", "the", "an", "each", "them", "therefore", "e.g.", ",", "a", "one", "on", "at",
                                    "by", "when", "such", "all", "this", "given", "its", "their"]
        else:
            irrelevant_determiners = ["der", "die", "das", "dem", "den", "ein", "eine", "des", "bei", "beim"]

        term_changed = False
        while nc_splitted[0].lower() in irrelevant_determiners:
            cleaned_nc = ' '.join([w for w in nc_splitted[1:]])
            nc_splitted = nc_splitted[1:]
            term_changed = True

            if len(nc_splitted) == 0:
                return None

        if not term_changed:
            cleaned_nc = nc
    except Exception as e:
        logging.info(f'Nc --- {nc}')
        logging.error(traceback.format_exc())

    return cleaned_nc.strip()

def advanced_lemmatizer(l_tok, language_flag):
    invalidcharacters = set(string.punctuation)
    if any(char in invalidcharacters for char in l_tok):
        return l_tok
    if check_if_word_is_abbreviation(l_tok):
        return l_tok
    if language_flag == "EN":
        new_doc = nlp_en(l_tok)
    else:
        new_doc = nlp_de(l_tok)
    return new_doc[-1].lemma_


def normalize_nc(nc, language_flag):
    if len(nc.split()) <= 1:
        return advanced_lemmatizer(nc, language_flag)

    cleaned_nc = re.sub(r"[\([{})\]]", "", nc)
    cleaned_nc = re.sub("(\w)([,|.])(\s)", r"\1 ", cleaned_nc)
    splitted_cleaned_nc = cleaned_nc.split()
    if len(splitted_cleaned_nc) <= 1:
        return advanced_lemmatizer(nc, language_flag)

    last_token = splitted_cleaned_nc[-1]
    splitted_cleaned_nc.pop()
    splitted_cleaned_nc.append(advanced_lemmatizer(last_token, language_flag))
    return " ".join(tok for tok in splitted_cleaned_nc)

def is_numeric(nc):
    
    for term in nc.split():
        term = re.sub(r'[^\w\s]','',term)
        if term.isnumeric():
            return False
    return True

def clean_normalized_word(nc):
    
    clean_words = []
    for term in nc.split():
        if term.lower() not in all_stopwords:
            clean_words.append(term)
    return " ".join(clean_words)

def get_shorter_text(phrase_1, phrase_2):
    
    if len(phrase_1) < len(phrase_2):
        return phrase_1
    else:
        return phrase_2

def remove_smallcase_words(noun_chunk):
    
    word_list = []
    for word in noun_chunk.split():
        last_char = word[0]
        if last_char.isupper():
            word_list.append(word)
                        
    return ' '.join(word_list)

def get_cleaned_noun_chunks(noun_chunks):
    
    noun_chunks = list(set(noun_chunks))
    remove_phrases = [] 
    phrases_len = len(noun_chunks)
    
    for idx_1 in range(phrases_len):

        phrase_1 = noun_chunks[idx_1]
        for idx_2 in range(idx_1 + 1, phrases_len):
            phrase_2 = noun_chunks[idx_2]

            if fuzz.ratio(phrase_1, phrase_2) > 85:
                remove_phrases.append(get_shorter_text(phrase_1, phrase_2))

    final_noun_chunks = list(set(noun_chunks) - set(remove_phrases))
    return final_noun_chunks
    

def extract_and_clean_ncs(sent, language_flag, representation_level): #TERM, TEXT
    language_flag = language_flag.upper()
    if representation_level == "TERM":
        return [(sent, normalize_nc(sent, language_flag))]
    else:
        noun_chunks_set_for_sent = set()
        if language_flag == "EN":
            doc = nlp_en(sent)
            determiners_list = ["each", "the", "an", "each", "them", "therefore", "e.g.", ",", "a", "one", "on", "at",
                                "by", "when", "such", "all", "this", "given", "its", "their", "as"]
        else:
            doc = nlp_de(sent)
            determiners_list = ["der", "die", "das", "dem", "den", "ein", "eine", "des", "bei", "beim"]
        for chunk in doc.noun_chunks:
            tmp_1 = chunk.text
            tmp_1 = tmp_1.strip("(")
            tmp_1 = tmp_1.strip(")")
            if len(tmp_1.split()) == 1 and tmp_1.strip().lower() in determiners_list:
                pass
            else:
                nc_with_removed_determiners = remove_determiners(tmp_1, language_flag)
                if nc_with_removed_determiners is not None and len(clean_string_from_puntuation_and_special_chars(nc_with_removed_determiners)) > 0:
                    noun_chunks_set_for_sent.add(nc_with_removed_determiners)

        composed_terms = set()
        for nc1 in noun_chunks_set_for_sent:
            for nc2 in noun_chunks_set_for_sent:
                if language_flag == "EN":
                    comp_term1 = nc1 + " of " + nc2
                    comp_term2 = nc1 + " and " + nc2
                    comp_term3 = nc1 + "'s " + nc2
                    if comp_term1 in sent:
                        composed_terms.add(comp_term1)
                    if comp_term2 in sent:
                        composed_terms.add(comp_term2)
                    if comp_term3 in sent:
                        composed_terms.add(comp_term3)
                else:
                    comp_term = nc1 + " und " + nc2
                    if comp_term in sent:
                        composed_terms.add(comp_term)
        found_terms = noun_chunks_set_for_sent.union(composed_terms)

        cleaned_terms = []
        for original_term in found_terms:
            
            if len(original_term.split()) < 4 and original_term.lower() not in all_stopwords and is_numeric(original_term):
                if len(original_term) > 0:
                    normalized_word = normalize_nc(original_term, language_flag)
                    normalized_word = re.sub(r'[^\w\s]','',normalized_word)
                    normalized_word = clean_normalized_word(normalized_word)

                    if language_flag == "DE" and len(normalized_word) > 1:
                        normalized_word = remove_smallcase_words(normalized_word)

                    if len(normalized_word) > 0 and is_numeric(normalized_word):
                        cleaned_terms.append(normalized_word)
                    
        cleaned_terms = get_cleaned_noun_chunks(cleaned_terms)
        return cleaned_terms