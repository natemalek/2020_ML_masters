# This file contains util functions for representing tweets using word
# embeddings
# Pipeline: Takes a list of tokens as input;
# extract_embeddings(tokens) returns those as a list of embeddings
# sum_pool_embeddings(embeddings) returns a sum-pooled embedding vector

import pandas as pd
import numpy as np
import nltk
import gensim

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


if __name__ == "__main__":
    embedding_filename = "C:/Users/natha/Documents/GoogleNews-vectors-negative300.bin"
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_filename, binary=True)

    train_filepath = "./data/cleaned_training.txt"
    test_filepath = "./data/cleaned_test.txt"
    dev_filepath = "./data/cleaned_dev.txt"
    paths = [train_filepath, test_filepath, dev_filepath]
    for filepath in paths:
        df = pd.read_csv(filepath, delimiter="\t")
        tweet_embeddings = []
        for tweet in df["Tweet"]:
            if type(tweet) != float:
                tweet_split = tweet.split()
                embedding_list = extract_embeddings(tweet_split, embedding_model)
                pooled_embedding = sum_pool_embeddings(embedding_list)
                tweet_embeddings.append(pooled_embedding)
            else:
                tweet_embeddings.append([0]*300)
        new_df = pd.DataFrame()
        new_df["Embedding"] = tweet_embeddings
        new_df["Valence score"] = df["Valence score"]
        new_filename = "./data/embeddings_"+filepath.split("_")[1].strip(".txt")+".pkl"
        new_df.to_pickle(new_filename)
    
    
    
    