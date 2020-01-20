# Utils for preprocessing
### Import all needed modules
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import re
import string
import os
import emoji
from pprint import pprint
import collections

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def import_file(filepath):
    '''
    Takes a filepath to a headerful tsv file and returns the data as a
    pandas df.
    '''

    df = pd.read_table(filepath)

    return df

def adapt_valence_scores(df):
    """
    Takes a dataframe and converts the valence scores present into a shortened number
    instead of the long description in the original file.

    :returns valence_list (list of valence scores as numbers)
    """
    valence_list = list()
    for index, row in df.iterrows():
        valence = row["Intensity Class"]
        valence = valence.replace(valence, valence[:2].replace(":", ""))
        valence_list.append(valence)
    return valence_list

class CleanText(BaseEstimator, TransformerMixin):
    """
    From https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27
    """
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)

    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')

    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)

    def to_lower(self, input_text):
        return input_text.lower()

    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whivtelist = ["n't", "not", "no"]
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords)
        return clean_X

def create_dataframe(clean_list, valence_scores, drop_row=None):
    """
    Adds the cleaned tweet list and the valence scores list together in one dataframe.
    If a drop row is defined, use this to drop a certain row with that index from the dataset
    """
    cleaned = pd.DataFrame(clean_list)
    cleaned['Valence score'] = adapt_valence_scores
    if drop_row != None:
        cleaned = cleaned.drop([drop_row])
    return cleaned

if __name__=="__main__":
   
