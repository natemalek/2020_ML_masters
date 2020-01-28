
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
Nathan

Text explaining what it does.

#### Which packages are needed?
text

#### How to run the script?
Which arguments in command line?
Structure of input, structure of output.
What should the output look like (so you know you've done it correctly)

### create_classifiers.py
??
