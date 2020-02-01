# This file contains util functions for representing tweets using word
# embeddings
# Pipeline: Takes a list of tokens as input;
# extract_embeddings(tokens) returns those as a list of embeddings
# sum_pool_embeddings(embeddings) returns a sum-pooled embedding vector

# To run: python tweet_embedding.py embedding_model_filepath embedding_filetype lexicon_meta_filepath data_filepath new_filepath
# eg: python tweet_embedding.py C:/users/natha/Documents/GoogleNews-vectors-negative300.bin G ./lexicon/English/lexicon_meta_data_English.txt ./data/cleaned/cleaned-Valence-reg-En-train.txt ./data/embeddings/embeddings-Valence-reg-En-train.pkl
# embedding_filetype: "G" or "A" for GoogleNews/AraVec, respectively


import pandas as pd
import numpy as np
import nltk
import gensim
import time
import sys
import clean_sentiment_lexicon

in_lexicon = 0
not_in_lexicon = 0

def extract_embeddings(tokens, embedding_model):
    '''
    Takes a list of tokens and returns a list of their word2vec embeddings, as 
    contained in an embedding model
    
    :param tokens: a list of tokens
    :param embedding_model: a gensim embedding model
    
    :returns embeddings: a list of word embeddings
    '''
    embeddings = []

    for word in tokens:
        if word in embedding_model:
            embeddings.append(embedding_model[word])
        else:
            embeddings.append([0]*300)

    return embeddings

def sum_pool_embeddings(embeddings):
    '''
    takes a list of word embeddings and returns a single embedding constructed
    from them using a sum-pooling operation.
    
    :param embeddings: a list of word embedding vectors
    
    :returns pooled_embedding: a single embedding vector
    '''
    pooled_embedding = []
    for i in range(len(embeddings[0])):
        dim_sum = 0
        for vector in embeddings:
            dim_sum += vector[i]
        pooled_embedding.append(dim_sum)
    return pooled_embedding

def check_lexicon(lexicon_dict):
    '''
    Asserts that every sub-dict in the lexicon dict has the same number of entries.
    
    :param lexicon_dict: a dict of dicts that is a sentiment lexicon. The structure
        is as follows: {word: {score_type: score}} ie {"happy": {"joy": 2.1, ...}, ...}
    
    :returns (True, length) or (False, length), where length is the size of the first sub-dict
    '''
    first = True
    same_length = True
    
    for key, value_dict in lexicon_dict.items():
        if first:
            l = len(value_dict)
            first = False
        else:
            if len(value_dict) != l:
                same_length = False
                break
    return same_length, l

def compute_lexicon_score(tweet, lexicon_dict, lex_subdict_length):
    '''
    Computes a sentiment lexicon score for a tweet by summing scores over the words
    in the tweet.
    
    :param tweet: a string of words
    :param lexicon_dict: a dict of dicts that is a sentiment lexicon. The structure
        is as follows: {word: {score_type: score}} ie {"happy": {"joy": 2.1, ...}, ...}
    
    :returns tweet_scores: a list of summed scores for each word in the tweet
    '''
    tweet_score_dict = dict()
    tweet_scores = list()
    global in_lexicon
    global not_in_lexicon
    
    tweet_populated = False
    for word in tweet.split(" "):
        if word in lexicon_dict:
            tweet_populated = True
            in_lexicon += 1
            for score_type, score in lexicon_dict[word].items():
                if score_type in tweet_score_dict:
                    tweet_score_dict[score_type] += np.float64(score)
                else:
                    tweet_score_dict[score_type] = np.float64(score)
        else:
            not_in_lexicon += 1
                    
    # Make tweet_scores list, ensuring consistent ordering of dimensions
    score_tuple_list = list(tweet_score_dict.items())
    score_tuple_list.sort()  # sort list of item tuples by key
    
    if tweet_populated:
        # if at least one word in lexicon_dict, append lexicond_dict scores.
        for t in score_tuple_list:
            tweet_scores.append(t[1])
    else:
        # otherwise, fill with appropriate number of 0s
        tweet_scores+=[0]*lex_subdict_length

    return tweet_scores

def collect_embeddings(filepath, lexicon, embedding_model):
    '''
    Takes a tsv filepaths with tweets and scores and creates a dataframe with tweets
    represented as embeddings and the same scores. Takes also a sentiment lexicon
    dictionary.
    
    :param filepath: the filepath to a tsv file
    :param lexicon: a dict of dicts representing a sentiment lexicon. Format:
        {word: {score_type: score, ...}, ...}
    :param embedding_model: a gensim word embedding model
    '''
    df = pd.read_csv(filepath, delimiter="\t")
    tweet_embeddings = []
    
    same_length, length = check_lexicon(lexicon)
    assert same_length, "sub-dicts in lexicon_dict must be all the same length"
    
    for tweet in df["Tweet"]:
        # sum_pooled embeddings
        if type(tweet) != float: #empty tweet after preprocessing
            tweet_split = tweet.split()
            embedding_list = extract_embeddings(tweet_split, embedding_model)
            pooled_embedding = sum_pool_embeddings(embedding_list)
        else:
            pooled_embedding = [0]*300
        
        # lexicon scores
        score_list = compute_lexicon_score(tweet, lexicon, length)
        pooled_embedding += score_list # concatenate with embeddings
        tweet_embeddings.append(pooled_embedding)
        
    new_df = pd.DataFrame()
    new_df["Embedding"] = tweet_embeddings
    new_df["Valence score"] = df["Valence score"]
    return new_df
    

if __name__ == "__main__":
    
    embedding_filename = sys.argv[1]
    embedding_filetype = sys.argv[2]
    lexicon_meta_filepath = sys.argv[3]
    data_filename = sys.argv[4]
    new_filename = sys.argv[5]
    
    start = time.time()

    if embedding_filetype == "G":
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_filename, binary=True)
    else:
        embedding_model = gensim.models.Word2Vec.load(embedding_filename)

    embedding_done = time.time()
    
    lexicon = clean_sentiment_lexicon.import_sentiment_lexicons(lexicon_meta_filepath)
    
    new_df = collect_embeddings(data_filename, lexicon, embedding_model)
    new_df.to_pickle(new_filename)
    
    '''
    paths = ["data/cleaned/cleaned-Valence-reg-Ar-train.txt", "data/cleaned/cleaned-Valence-reg-Ar-test.txt", "data/cleaned/cleaned-Valence-reg-Ar-dev.txt"]
    for filepath in paths:
        new_filepath = "./data/embeddings/embeddings-"+'-'.join(filepath.split(".")[0].split("-")[1:])+".pkl"
        
    end = time.time()
    print(f"Total time: {end-start}")
    print(f"Time after embedding: {end-embedding_done}")
    print(f"Total tokens not in lexicon: {not_in_lexicon}; total words in lexicon: {in_lexicon}")
    '''
    
    
    