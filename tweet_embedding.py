# This file contains util functions for representing tweets using word
# embeddings
# Pipeline: Takes a list of tokens as input;
# extract_embeddings(tokens) returns those as a list of embeddings
# sum_pool_embeddings(embeddings) returns a sum-pooled embedding vector

# To run: python tweet_embedding.py embedding_model_filepath embedding_filetype lexicon_meta_filepath
# embedding_filetype: "G" or "A" for GoogleNews/AraVec, respectively

# Jan 23 morning: 
# in Arabic training set: 10076 tokens in file successfully embedded, 1103 tokens in file not in embedding model
# in English training set: 9420 embedded, 721 missed

# Jan 24 morning, lexicon coverage:
# English: 9279 in lexica, 862 missed



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

def compute_lexicon_score(tweet, lexicon_dict):
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
    for word in tweet.split(" "):
        if word in lexicon_dict:
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
    score_tuple_list.sort()
    for t in score_tuple_list: # sort list of item tuples by key
        tweet_scores.append(t[1])
    
    return tweet_scores

def collect_embeddings(filepath, new_filepath, lexicon):
    '''
    Takes a tsv filepaths with tweets and scores and writes a new file with tweets
    represented as embeddings and the same scores. Takes also a sentiment lexicon
    dictionary.
    
    :param filepath: the filepath to a tsv file
    :param new_filepath: filepath to new embeddings file
    :param lexicon: a dict of dicts representing a sentiment lexicon. Format:
        {word: {score_type: score, ...}, ...}
    '''
    df = pd.read_csv(filepath, delimiter="\t")
    tweet_embeddings = []
    for tweet in df["Tweet"]:
        # sum_pooled embeddings
        if type(tweet) != float: #empty tweet after preprocessing
            tweet_split = tweet.split()
            embedding_list = extract_embeddings(tweet_split, embedding_model)
            pooled_embedding = sum_pool_embeddings(embedding_list)
        else:
            pooled_embedding = [0]*300
        
        # lexicon scores
        score_list = compute_lexicon_score(tweet, lexicon)
        
        pooled_embedding += score_list # concatenate with embeddings
        tweet_embeddings.append(pooled_embedding)
    new_df = pd.DataFrame()
    new_df["Embedding"] = tweet_embeddings
    new_df["Valence score"] = df["Valence score"]
    new_df.to_pickle(new_filepath)
    

if __name__ == "__main__":
    
    embedding_filename = sys.argv[1]
    embedding_filetype = sys.argv[2]
    lexicon_meta_filepath = sys.argv[3]
    
    start = time.time()

    if embedding_filetype == "G":
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_filename, binary=True)
    else:
        embedding_model = gensim.models.Word2Vec.load(embedding_filename)

    embedding_done = time.time()
    
    lexicon = clean_sentiment_lexicon.import_sentiment_lexicons(lexicon_meta_filepath)
    
    '''
    Arabic_train_filepath = "./data/cleaned/cleaned-Valence-reg-Ar-train.txt"
    
    train_filepath = "./data/cleaned/cleaned-EI-reg-En-anger-train.txt"
    test_filepath = "./data/cleaned/cleaned-EI-reg-En-anger-test.txt"
    dev_filepath = "./data/cleaned/cleaned-EI-reg-En-anger-dev.txt"
    
    paths = [dev_filepath, test_filepath, train_filepath]
    '''
    paths = ["./data/cleaned/cleaned-Valence-reg-En-train.txt", "./data/cleaned/cleaned-Valence-reg-En-test.txt", "./data/cleaned/cleaned-Valence-reg-En-dev.txt"]
    for filepath in paths:
        new_filepath = "./data/embeddings/embeddings-"+'-'.join(filepath.strip(".txt").split("-")[1:])+".pkl"
        collect_embeddings(filepath, new_filepath, lexicon)
    
    end = time.time()
    print(f"Total time: {end-start}")
    print(f"Time after embedding: {end-embedding_done}")
    print(f"Total tokens not in lexicon: {not_in_lexicon}; total words in lexicon: {in_lexicon}")
    
    
    