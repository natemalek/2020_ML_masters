
# vaguely_ML_masters

## notes, remove later
install emoji package

gensim arabic word embeddings:
    Citation: Abu Bakr Soliman, Kareem Eisa, and Samhaa R. El-Beltagy, “AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP”, in proceedings of the 3rd International Conference on Arabic Computational Linguistics (ACLing 2017), Dubai, UAE, 2017.
    Download from: https://github.com/bakrianoo/aravec

Arabic sentiment lexica: http://saifmohammad.com/WebPages/ArabicSA.html

## Introduction

This project is designed to tackle the SemEval 2018 Task 1: Affect in Tweets. It includes files which preprocess the English and Arabic files provided for this task (https://competitions.codalab.org/competitions/17751#learn_the_details-datasets), process sentiment lexica for both languages (http://sentiment.nrc.ca/lexicons-for-research/ and http://www.saifmohammad.com/WebPages/ArabicSA.html), and perform and evaluate xgboost regression or ordinal classification.

The "pipeline" for applying these functions is roughly as follows:
preprocessing.py is applied to the raw SemEval 2018 Task 1 English and Arabic data.
clean_sentiment_lexicon.py is applied to the NRC sentiment lexicons listed above.
regression_main.py is applied to the preprocessed/cleaned data, with the evaluation of the system printed in results.txt.

The files tweet_embedding.py and regressors.py are primarily designed to be called by regression_main.py, but can be run independently on appropriate files (for testing of intermediate steps, for instance).

## Getting ready

Python version 3.7.3 was used.

### Needed packages
The following packages are needed for running this project. You might need to install them.

- pandas
- re  
- string
- os
- emoji  
- collections
- glob
- sklearn
- scipy
- nltk
- gensim
- xgboost

### Needed files
The following files were used in this project.

- Lexica Arabic and English
- meta_data files for arabic and english lexica
=======
- List of Arabic stopwords imported from [LINK TO THE ARABIC STOPWORDS MODULE]
- Lexica Arabic and English (http://sentiment.nrc.ca/lexicons-for-research/ and http://www.saifmohammad.com/WebPages/ArabicSA.html)
- English pre-trained word embeddings: GoogleNews (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
- Arabic pre-trained word embeddings: AraVec (https://github.com/bakrianoo/aravec/tree/master/AraVec%202.0); specifically, the 300-dimension Twitter SkipGram embeddings were utilized.


### Folder structure
The files are sorted and saved in the following manner

## Running the scripts
In this part we will walk you through the steps of each notebook.


### preprocessing.py
The preprocessing file takes filepath to a TSV file and returns a cleaned TSV file. The cleaning process includes converting the valence column to a shortened format, in the sense that it takes only the number score, instead of including the entire description (adapt_valence_scores). In addition, the tweets are cleaned and prepared for further processing and model building. The Clean_Text class is different for English and Arabic tweets. For both languages, the hashtags are removed, as well as repeating characters, mentions, urls, punctuation and digits. Additionally, emojis are converted to one word characters and all characters are set to lowercase. For English, specifically, stopwords are removed (with the exception of the whitelisted words). These stopwords are taken from the nltk module. For Arabic, the stopwords are imported from [LINK TO THE ARABIC STOPWORDS MODULE], and removed in a similar manner as for English. For Arabic, there are several additional steps in cleaning the text such as removing tashkeel (optional accents in the Arabic text) as well as further character cleaning by replacing certain characters with others or nothing (This code is adapted from https://github.com/bakrianoo/aravec). The data for each file in the raw data map is then written to a folder called "cleaned/" with fitting names.

#### Which packages are needed?
- pandas
- re  
- string
- os
- emoji  
- collections
- glob
- sklearn
- nltk

Additionally, you have to download a list of Arabic stopwords.

#### How to run the script?
Run from command line: python preprocessing.py input_directory output_directory
input_directory: the path to a directory containing files to be cleaned (and only these files)
output_directory: the path to a directory where new files will be stored

The input should be structured as a TSV file, ending in '.txt' in order make it work. In the input file the columns should have the following names in order to work "Intensity Class" or "Intensity Score". In the basename, there should be "En" or "Ar", indicating the language of the tweets in order for the script to run correctly.

The output is a .txt file that should look like a TSV file, with the index in the first column, the tweet in the second and the valence or intensity score in the third column.

### clean_sentiment_lexicon.py
Quirine

This script contains a collection of functions which clean any amount of sentiment lexica and joins them together in one dictionary. The main function is import_sentiment_lexicons(). This takes the file path to a file containing meta data of the lexicons (see more under Needed files).

#### Which packages are needed?
text

#### How to run the script?
Which arguments in command line?
Structure of input, structure of output.
What should the output look like (so you know you've done it correctly)

### tweet_embedding.py
This file contains util functions for representing tweets using word embeddings.
It is used primarily as utility functions for regression_main.py, but can be run on its own to output a .pkl file
containing word embeddings.

Pipeline: Takes a pandas df .tsv file as input, and outputs a .pkl file with embeddings

#### Which packages are needed?
Required packages: pandas, numpy, nltk, gensim, sys, clean_sentiment_lexicon
Required files:
* sentiment lexicon (see clean_sentiment_lexicon documentation)
* Embedding model(s): GoogleNews (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), AraVec (https://github.com/bakrianoo/aravec). Filepath to embedding model is passed as a command line argument.

#### How to run the script?
Command: "python tweet_embedding.py embedding_model_filepath embedding_filetype lexicon_meta_filepath data_filepath new_filepath"
* embedding_filetype: "G" or "A" for GoogleNews/AraVec, respectively
* lexicon_meta_filepath: see clean_sentiment_lexicon documentation for what this file should look like

### regressors.py
This file contains functions for creating, applying, and evaluating xgboost regression models for regression
and ordinal classification.
It is used primarily as utility functions for regression_main.py, but can be run on its own to read train and test .pkl files
containing embeddings and labels and output the evaluation of the model to standard output.

#### Which packages are needed?
Required packages: pandas, sklearn, sys, scipy, xgboost

#### How to run the script?
Command: "python regressors.py train_file.pkl test_file.pkl"
train_file.pkl and test_file.pkl structured as is the output of tweet_embedding.py: a pandas df with embeddings in col 1 and
labels in col 2.

### regression_main.py
This file calls tweet_embedding.py and regressors.py to perform and evaluate xgboost regression or ordinal classification
on a pandas df .tsv file, and to output the evaluation into results.txt.

The input must be preprocessed files with one column called "Tweets" and another called "Valence Score" (as created by
preprocessing.py, see above).

#### Which packages are needed?
Required python packages: gensim, os, sys

#### How to run the script?
Command: "python regression_main.py embedding_filename embedding_filetype lexicon_filename train_filename test_filename model_type"
embedding_filename: The path to a file containing pretrained word embeddings (either GoogleNews Word2Vec or AraVec).
embedding_filetype: 'A' or 'G', for AraVec or GoogleNews, respectively
lexicon_filename: The path to the lexicon metafile. Details on this file can be found in clean_sentiment_lexicon.py documentation.
train_filename, test_filename: The paths to train and test data, structured as .tsv files with "Tweet" and "Valence Score" columns
model_type: 'oc' or 'reg', for Ordinal Classification and Regression, respectively.

## References
