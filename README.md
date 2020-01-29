
# vaguely_ML_masters

## notes, remove later
install emoji package

gensim arabic word embeddings:
    Citation: Abu Bakr Soliman, Kareem Eisa, and Samhaa R. El-Beltagy, “AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP”, in proceedings of the 3rd International Conference on Arabic Computational Linguistics (ACLing 2017), Dubai, UAE, 2017.
    Download from: https://github.com/bakrianoo/aravec

Arabic sentiment lexica: http://saifmohammad.com/WebPages/ArabicSA.html

## Introduction

Short introduction to project

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
- nltk

### Needed files
The following files were used in this project.

- List of Arabic stopwords imported from [LINK TO THE ARABIC STOPWORDS MODULE]
- Lexica Arabic and English


### Folder structure
The files are sorted and saved in the following manner

## Running the scripts
In this part we will walk you through the steps of each notebook.

### main.py
This file takes two preprocessed (cleaned) data files (one for training, one for testing), and outputs the results of the classification system on the data.
Specifically, this appends an evaluation score (Pearson r) to the file "results.txt".

#### How to run the script?
"python main.py embedding_filename embedding_filetype lexicon_meta_filepath train_filename test_filename"
* embedding_filetype: "G" or "A" for GoogleNews/AraVec, respectively
* lexicon_meta_filepath: see clean_sentiment_lexicon documentation for what this file should look like
* train_filename & test_filename: .tsv files containing pandas df objects (as produced, for example, by preprocessing.py)
* eg: "python main.py C:/users/natha/Documents/GoogleNews-vectors-negative300.bin G ./lexicon/English/lexicon_meta_data_English.txt ./data/cleaned/cleaned-Valence-reg-En-train.txt ./data/embeddings/embeddings-Valence-reg-En-train.pkl"

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
The script can be run by opening the file in Spyder and running it here. [IS THIS CORRECT?]

The input should be structured as a TSV file, ending in '.txt' in order make it work. In the input file the columns should have the following names in order to work "Intensity Class" or "Intensity Score". In the basename, there should be "En" or "Ar", indicating the language of the tweets in order for the script to run correctly.

The output is a .txt file that should look like a TSV file, with the index in the first column, the tweet in the second and the valence or intensity score in the third column.

### clean_sentiment_lexicon.py
Quirine

Text explaining what it does.

#### Which packages are needed?
text

#### How to run the script?
Which arguments in command line?
Structure of input, structure of output.
What should the output look like (so you know you've done it correctly)

### tweet_embedding.py
This file contains util functions for representing tweets using word embeddings.
It is used primarily as utility functions for main.py, but can be run on its own to output a .pkl file
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
eg: "python tweet_embedding.py C:/users/natha/Documents/GoogleNews-vectors-negative300.bin G ./lexicon/English/lexicon_meta_data_English.txt ./data/cleaned/cleaned-Valence-reg-En-train.txt ./data/embeddings/embeddings-Valence-reg-En-train.pkl"

### create_classifiers.py
??
