# This file contains util functions for representing tweets using word
# embeddings
# Pipeline: Takes a list of tokens as input;
# extract_embeddings(tokens) returns those as a list of embeddings
# sum_pool_embeddings(embeddings) returns a sum-pooled embedding vector

import pandas as pd
import numpy as np
import nltk
import gensim
embedding_file = "C:\Users\natha\Documents\GoogleNews-vectors-negative300.bin"
def extract_embeddings(tokens, embedding_filename=embedding_file):
    '''
    Takes a list of tokens and returns a list of their word2vec embeddings, as 
    contained in a model embedding file
    
    :param tokens: a list of tokens
    :param embedding_filename: path to a .bin word2vec format word embedding file
    
    :returns embeddings: a list of word embeddings
    '''
    embeddings = []

    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_filename, binary=True)
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

    
    
    