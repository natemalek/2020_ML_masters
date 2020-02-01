### This file takes a directory as an argument and preprocesses all files in that directory.
### It also takes a second directory for the storage of new, processed files.
# to run: python preprocessing.py input_directory output_directory
# eg. python preprocessing.py data/raw/ data/cleaned/

# Utils for preprocessing
### Import all needed modules
import pandas as pd
import re
import string
import os
import emoji
import collections
import glob
import sys

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

def get_stopwords(filepath):
    """
    Takes a filepath to a txt file and returns the data as a list of strings
    """
    stopwords = list()
    with open(filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.replace("\n", "")
            stopwords.append(line)
    return stopwords

def adapt_valence_scores(df, column_name):
    """
    Takes a dataframe and converts the valence scores present into a shortened number
    instead of the long description in the original file.

    :returns valence_list (list of valence scores as numbers)
    """
    valence_list = list()
    for index, row in df.iterrows():
        valence = row[column_name]
        if column_name == 'Intensity Class':   # If oc then need to alter valence representation
            valence = valence.replace(valence, valence[:2].replace(":", ""))
        valence_list.append(valence)
    return valence_list

class CleanText(BaseEstimator, TransformerMixin):
    """
    Takes a tweet and returns a cleaned version. Several functions are in play
    for English and Arabic respectively, which are annotated in the function
    descriptions.

    (Format adapted from https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27)
    (Functions for Arabic were adapted from the processing section of
    https://github.com/bakrianoo/aravec for Arabic word embeddings)

    :returns two cleantext functions, one for English and one for Arabic
    """
    def __init__(self, language = ''):
        self._language = language

    def remove_hashtags(self, input_text):
        return re.sub(r"#", "", input_text)

    def remove_repeating_char(self, input_text):
        return re.sub(r'(.)\1+', r'\1\1', input_text)
        #keep 2 repeating characters in order to differentiate them from the
        #normal format

    def remove_tashkeel(self, input_text):
        """
        Removes optional accents from the Arabic texts.
        """

        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        return re.sub(p_tashkeel,"", input_text)

    def clean_char(self, input_text):
        input_text = input_text.replace('وو', 'و').replace('يي', 'ي').replace('اا', 'ا')

        search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
        replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']

        for i in range(0, len(search)):
            input_text = input_text.replace(search[i], replace[i])

        return input_text

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

    def remove_stopwords_english(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
        return " ".join(clean_words)

    def remove_stopwords_arabic(self, input_text):
        stopwords_list = get_stopwords("utilities/arabic-stop-words-master/list.txt")
        words = input_text.split()
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1]
        return " ".join(clean_words)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        if self._language == "En":
            ct = X.apply(self.remove_repeating_char).apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords_english)
            return ct
        elif self._language == "Ar":
            ct = X.apply(self.remove_hashtags).apply(self.remove_tashkeel).apply(self.clean_char).apply(self.remove_repeating_char).apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.remove_stopwords_arabic)
            return ct

if __name__ == '__main__':
    
    folder_path = sys.argv[1]
    out_folder_path = sys.argv[2]
    
# Get all files in folder to be cleaned
    raw_filepaths = glob.glob(folder_path + '*')
    # Loop over raw files
    for filepath in raw_filepaths:
        # Get basename
        basename = os.path.basename(filepath)
        basename_ = basename.split('-') # Get list of splitted basename
        # If a file is not a txt file, then print the file path, and skip to next
        if basename_[-1][-4:] != '.txt':
            print('Did not clean', filepath + '. Item did not end in .txt.')
            continue
        # Read in df
        df = pd.read_table(filepath)

        # Get valence score (column name is different for type of classification)
        if basename_[2] == 'oc':
            val_col = "Intensity Class"
        elif basename_[2] == 'reg':
            val_col = "Intensity Score"
        else:
            print('Classification type not found. basename_[2]', basename_[2])
        # Get valence list from df
        valence_list = adapt_valence_scores(df, val_col)

        # If data is English
        if basename_[3] == 'En':
            ct = CleanText(language='En')
            clean_ = ct.fit_transform(df.Tweet)

        # If data is Arabic
        elif basename_[3] == 'Ar':
            ct = CleanText(language='Ar')
            clean_ = ct.fit_transform(df.Tweet)

        # Fill empty cells with '[no_text]', and print how many there are
        empty_clean = clean_ == ''
        if clean_[empty_clean].count() > 0:
            print(f'{clean_[empty_clean].count()} records have no words left after text cleaning in {basename}')
            clean_.loc[empty_clean] = '[no_text]'

        # Create new df of cleaned data
        df_cleaned = pd.DataFrame(clean_)
        # Add valence score to list
        df_cleaned['Valence score'] = valence_list

        # Drop rows with no tweet text
        df_cleaned = df_cleaned[df_cleaned['Tweet'] != '[no_text]']
        df_cleaned.reset_index(drop=True, inplace=True)

        # Write to csv
        if len(basename_) == 5:
            outpath = out_folder_path + '-'.join(['cleaned', basename_[1], basename_[2], basename_[3], basename_[-1]])
        elif len(basename_) == 6:
            outpath = out_folder_path + '-'.join(['cleaned', basename_[1], basename_[2], basename_[3], basename_[4], basename_[-1]])
        else:
            print('something went wrong with basename length. It is not 5 or 6 elements long.')
        # For example "data/cleaned/cleaned_Valence_oc_En_train.txt'
        df_cleaned.to_csv(outpath, sep="\t")
