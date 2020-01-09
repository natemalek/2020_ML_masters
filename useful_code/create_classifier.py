### This file contains functions which build a logistic regression, naive Bayes, and/or svm model based on a tsv file, 
### applies the model to a test set, and outputs the predictions to a tsv file with two columns (token and prediction).
### Note that this code is interactive: user will be prompted for, amongst other things, an outputfile path.

# Command line call: python basic_system.py trainingfile test_data_file

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted
from preprocess import import_conll, extract_tokens
import sys
import nltk
import gensim
import numpy as np

def extract_embeddings(tokens, embedding_filename='../GoogleNews-vectors-negative300.bin'):
    '''
    Takes a list of tokens and returns a list of their word2vec embeddings, as contained in a model embedding file
    
    :param tokens: a list of tokens
    :param embedding_filename: a .bin word2vec format word embedding file
    
    :returns embeddings: a list of word embeddings
    '''
    embeddings = []
    #non_set = set()
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_filename, binary=True)
    for word in tokens:
        if word in embedding_model:
            embeddings.append(embedding_model[word])
        else:
            #non_set.add(word)
            embeddings.append([0]*300)
    #print(non_set)
    return embeddings
    
    
def extract_pos_tags(tokens):
    '''
    Takes a list of tokens and returns a list of pos tags done by nltk pos tagger.
    
    :param tokens: a list of strings (language tokens)
    
    :returns tags: a list of pos_tags
    '''
    tag_tuples = nltk.pos_tag(tokens)
    tags = []
    for token, tag in tag_tuples:
        tags.append(tag)
    return tags

def extract_capitalization(tokens):
    '''
    Takes a list of tokens and returns a list of capitalization classification,
        using a ternary system (follows-period, non-capitalized, doesn't-follow-period-is-capitalized)
    
    :param tokens: a list of strings (word tokens)
    
    :returns cap_tags: a list of "Cap", "NC" (non-cap), "FP" (follows period) tags
    '''
    cap_tags = []
    first_tag = tag_token_capital(tokens[0])
    cap_tags.append(first_tag)
    for i in range(1, len(tokens)):
        cap_tags.append(tag_token_capital(tokens[i], tokens[i-1]))
    return cap_tags
    
def tag_token_capital(token, prev_token=None):
    '''
    Takes a token and its preceding token and returns a tag in a ternary capitalization
    system ("Cap", "NC" (non-cap), or "FP" (follows period)). First token in the list
    is assumed to be "FP"-tagged.
    
    :param prev_token: previous token
    :param token: token to be tagged for capitalization
    
    :returns cap_tag: the capitalization tag
    '''
    if prev_token == "." or prev_token == None:
        tag = "FP"
    else:
        if token[0].isupper():
            tag = "Cap"
        else:
            tag = "NC"
    return tag
    
def extract_features_and_labels(trainingfile, token_index=0, ne_label_index=1, 
                                feature_list=[("embedding",extract_embeddings), 
                                              ("pos_tag",extract_pos_tags), 
                                              ("cap_tag",extract_capitalization)]):
    '''
    Takes a file with training data and returns a list of feature dictionaries 
        and a list of ne_labels.
    Features included: token, pos_tag, cap_tag
    
    :param trainingfile: a string containing the path to the training data file
    :param token_index: the index of the column containing the tokens in the 
        training data file
    :param ne_label_index: the index of the column containing the ne_labels in 
        the training data file
    :param feature_list: a list of tuples representing the features to extract and the functions
        used to extract them.
    
    :returns features, ne_labels
    '''
    training_data = import_conll(trainingfile, token_col=token_index, label_col=ne_label_index)
    labels = training_data["labels"]
    tokens = training_data["tokens"]
    data = dict() # This will contain feat_name:feat_list pairs, ie {"token":["example","tokenized","sentence"],...}
    for feat, function in feature_list:
        if feat == "token":
            data["token"] = tokens
        else:
            data[feat] = function(tokens)
    
    features = []
    for i in range(len(tokens)):
        feature_dict = dict()
        for feat_name, feat_list in data.items():
            feature_dict[feat_name] = feat_list[i]
        features.append(feature_dict)
    
    return features, labels

def vectorize_features(feature_dicts, vec=False):
    '''
    Takes a list of feature dicts and returns them in vectorized form. Operates as a simple
    vectorizer if there are only simple features, but if 'embedding' is a feature than operates
    by extracting the embedding vectors and concatenating them with the vectorized features.
    
    :param feature_dicts: a list of feature dicts
    :param vec: a fitted vectorizer (if none provided, makes a new one)
    
    :returns vec_data: data represented as vectors
    :returns vec: the vectorizer fitted in the process
    '''
    
    embedding_list = []
    if 'embedding' in feature_dicts[0]:
        if 'token' in feature_dicts[0]:
            raise FeatureError("Error: Cannot use both tokens and embeddings. Program will abort.") 
        for feat_dict in feature_dicts:
            embedding = feat_dict.pop("embedding")
            embedding_list.append(embedding)

    if 'token' in feature_dicts[0]:
        # If using tokens, need to approach vectorizing differently (no .toarray())
        if not vec:
            # If no vectorizer provided, make one and fit_transform
            vec = DictVectorizer()
            feat_vecs = vec.fit_transform(feature_dicts)
        else:
            # If vectorizer provided, just transform
            feat_vecs = vec.transform(feature_dicts)
    else:
        if not vec:
            # If no vectorizer provided, make one and fit_transform
            vec = DictVectorizer()
            feat_vecs = vec.fit_transform(feature_dicts).toarray()
        else:
            # If vectorizer provided, just transform
            feat_vecs = vec.transform(feature_dicts).toarray()
    
    if embedding_list:
        feat_vecs = np.array(feat_vecs)
        embedding_list = np.array(embedding_list)
        vec_data = []
        for i in range(len(feature_dicts)):
            representation = list(feat_vecs[i]) + list(embedding_list[i])
            vec_data.append(representation)
    else:
        vec_data = feat_vecs
    
    return vec_data, vec

def create_logit_classifier(train_features, train_labels):
    '''
    Creates a logistic regression classifier for the given input features and 
    output labels.
    
    :param train_features: a list of dictionaries containing features for each
        item in the training data.
    :param train_labels: a list of classifications from the training data
    
    :returns model: a logistic regression model, and
             vec: the DictVectorizer
    '''
    
    vec_data, vec = vectorize_features(train_features)
    
    model = LogisticRegression().fit(vec_data, train_labels)
    
    return model, vec

def create_svm_classifier(train_features, train_labels):
    '''
    Creates an SVM classifier for the given input features and output labels.
    
    :param train_features: a list of dictionaries containing features for each
        item in the training data
    :param train_labels: a list of classifications from the training data
    
    :returns model: a logistic regression model, and 
             vec: the DictVectorizer
    '''
    
    vec_data, vec = vectorize_features(train_features)

    model = LinearSVC().fit(vec_data, train_labels)
    
    return model, vec

def create_nb_classifier(train_features, train_labels):
    '''
    Creates a logistic regression classifier for the given input features and 
    output labels.
    
    :param train_features: a list of dictionaries containing features for each
        item in the training data
    :param train_labels: a list of classifications from the training data
    
    :returns model: a logistic regression model, and 
               vec: the DictVectorizer
    '''
    
    vec_data, vec = vectorize_features(train_features)
    
    model = MultinomialNB().fit(vec_data, train_labels)
    
    return model, vec

def classify_data(model, vectorizer, test_data, outputfile):
    '''
    Makes predictions on test_data based on input model (using input vectorizer
    to transform the test_data) and outputs those predictions to outputfile.
    outputfile will be formatted as a .tsv, with the features in test_data
    in the first columns and the predictions in the final column.
    
    :param model: a sklearn model object
    :param vectorizer: a DictVectorizer object, fitted to the data model is
        trained on
    :param test_data: a list of feature dictionaries
    :param outputfile: the path to the desired output file
    
    :returns predictions: a list of the predictions made by model on test_data
    '''
    vec_data = vectorize_features(test_data, vec=vectorizer)[0]
    predictions = model.predict(vec_data)
    
    with open(outputfile, "a+") as outfile:
        for i in range(len(predictions)):
            feature_dict = test_data[i]
            if "token" in feature_dict.keys():
                outstring = str(feature_dict["token"]) #write token first
            else:
                outstring = "N/A"
            outstring += "\t" + str(predictions[i])
            for feature_name, feature in feature_dict.items():
                if feature_name != "token":
                    outstring += ("\t" + str(feature))
            outstring += "\n"
            outfile.write(outstring)

    return predictions

def main():
    # parse command line arguments python basic_system.py trainingfile test_data_file
    trainingfile = sys.argv[1]
    test_data_file = sys.argv[2]
    
    token_index = 0
    label_index = 1
    columns = input("Does the training file contain tokens in col 0 and labels in col 1? y/n: ")
    if columns == "n":
        custom_gold = True
        token_index = int(input("Enter the index of the token column: "))
        label_index = int(input("Enter the index of the label column: "))

    # extract feats and labels
    inp = input("Do you want to use default features (embeddings, pos_tags, cap_tags)? y/n: ")
    if inp == "n":
        feat_list = []
        function_list = [("token", True), ("embedding",extract_embeddings), ("pos_tag",extract_pos_tags), 
                         ("cap_tag", extract_capitalization)] # feats and their functions; token is a special case
        feat_inp = input("Enter a space-separated list of numbers corresponding to features to use:\n"
                         + "0. tokens\n"
                         + "1. embeddings\n"
                         + "2. pos_tags\n"
                         + "3. cap_tags\n"
                         + "Your input: ")
        for entry in feat_inp.split(' '):
            feat_list.append(function_list[int(entry)])
    else:
        feat_list = [("embedding",extract_embeddings), ("pos_tag",extract_pos_tags), 
                     ("cap_tag", extract_capitalization)]
    systems = input("Enter a space-separated list of classifiers to use ('logit' for Logistic regression, \
'nb' for Naive Bayes, and 'svm' for support vector machine): ").split(" ")
    columns = input("Does the test file contain tokens in col 0 and labels in col 1? y/n: ")
    if columns == "n":
        token_index = int(input("Enter the index of the token column: "))
        label_index = int(input("Enter the index of the label column: "))
    
    print("Extracting features from training file...")
    features, labels = extract_features_and_labels(trainingfile, token_index, label_index, feat_list)
    
    # create models
    print("Features successfully extracted. Creating model(s)...")
    models = []
    for entry in systems:
        if entry == "logit":
            model, vec = create_logit_classifier(features, labels)
            models.append(("logit", model, vec))
        elif entry == "nb":
            model, vec = create_nb_classifier(features, labels)
            models.append(("nb", model, vec))
        elif entry == "svm":
            model, vec = create_svm_classifier(features, labels)
            models.append(("svm", model, vec))
        else:
            print(f"Entry {entry} is invalid.")
    
    # get test data features
    token_index = 0
    label_index = 1
    print("Models successfully created. Extracting features from test data...")
    test_data = extract_features_and_labels(test_data_file, token_index, label_index, feat_list)[0]
    
    # create and write predictions to outpufile for each model
    print("Predicting labels and writing to outfile(s)...")
    for system, model, vec in models:
        outputfile = input(f"Enter the path to an outputfile for {system} output: ")
        classify_data(model, vec, test_data, outputfile)

if __name__ == '__main__':
    main()
    