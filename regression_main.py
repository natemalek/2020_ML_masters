import regressors
import tweet_embedding
import clean_sentiment_lexicon
import gensim
import sys
import os

# to run: python regression_main.py embedding_filename embedding_filetype lexicon_filename train_filename test_filename model_type
# embeddings_filetype: 'A' or 'G' for AraVec, GoogleNews Word2Vec, respectively
# model_type: 'reg' or 'oc', for regression or ordinal classification, respectively

def main():
    '''
    Runs the system, starting from cleaned data files. Imports embedding model,
    calls tweet_embeddings functions to attain df of embeddings, passes that
    df to create_classifiers functions.
    
    Appends results to "results.txt" (creates this file with headers if it doesn't yet exist)
    '''
    embedding_filename = sys.argv[1]
    embedding_filetype = sys.argv[2]
    lexicon_meta_filepath = sys.argv[3]
    train_path = sys.argv[4]
    test_path = sys.argv[5]
    model_type = sys.argv[6]
    
    results_file = "results.txt"
    if not os.path.exists('results.txt'):
        with open(results_file, "a+") as outfile:
            outfile.write("File\tPearson_r\n")

    if embedding_filetype == "G":
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_filename, binary=True)
    elif embedding_filetype == "A":
        embedding_model = gensim.models.Word2Vec.load(embedding_filename)
    else:
        print("Invalid embedding_filetype: Must be 'A' or 'G'. Aborting.")
        return
    
    lexicon = clean_sentiment_lexicon.import_sentiment_lexicons(lexicon_meta_filepath)
    
    df_train = tweet_embedding.collect_embeddings(train_path, lexicon, embedding_model)
    df_test = tweet_embedding.collect_embeddings(test_path, lexicon, embedding_model)
    if model_type == "reg":
        xgb_score_test, xgb_test_p_value = regressors.basic_xgboost_model(df_train, df_test)
    elif model_type == "oc":
        xgb_score_test, xgb_test_p_value = regressors.classification_xgboost_model(df_train, df_test)
    with open(results_file, "a+") as outfile:
        outfile.write(f"{train_path}\t{xgb_score_test}\n")

main()