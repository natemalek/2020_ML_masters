# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
from collections import defaultdict

filepath = './lexicon/English/lexicon_meta_data_English.txt'

def read_meta_file(filepath):
    meta_data = []
    with open(filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Remove \n at end of line
            if line.endswith('\n'):
                line = line[:-1]
            # Skip header line
            if line.startswith('index'):
                continue
            # Split lines on tab
            split_line = line.split("\t")
            # Change indexes from string to integer type
            split_line[2] = int(split_line[2])
            split_line[3] = int(split_line[3])
            split_line[4] = int(split_line[4])
            meta_data.append(split_line)
    return meta_data
            
def term_score_emotion_to_dict(meta_data_list):
    # Get data from meta_data_list
    index = meta_data_list[0]
    filename = meta_data_list[1]
    word_index = meta_data_list[2] 
    score_index = meta_data_list[3]
    emotion_index = meta_data_list[4]

    # Set dictionary
    word_dict = dict() # dict of dicts; {word1: {emotion1:score}, word2: {emotion1:score}}
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Remove \n at end of line
            if line.endswith('\n'):
                line = line[:-1]
            # Split lines on tab
            split_line = line.split("\t")
            # If line is empty or is header, skip to next
            if split_line[0] == '\n' or split_line[0] == 'term':
                continue
            # Set variables
            word = split_line[word_index]
            score = split_line[score_index]
            # If emotion index is given, the emotion + the index corresponding to the file
            # is the key of the dictionary for each word
            if emotion_index != -1:
                emotion = split_line[emotion_index] + index
            # If the emotion is not given then the emotion name is replaced by 'score'
            elif emotion_index == -1:
                emotion = 'score' + index
            # {word1: {emotion1:score, emotion2:score}, word2: {emotion1:score}}
            if word in word_dict:
                word_dict[word][emotion] = score
            else:
                word_dict[word] = dict()
                word_dict[word][emotion] = score
    return word_dict
 
def combine_dicts(list_of_dicts):
    main_dict = dict()
    for lexicon_dict in list_of_dicts:
        for word, inner_dict in lexicon_dict.items():
            if word not in main_dict:
                main_dict[word] = dict([('positive3', 0), ('score2', 0), 
                         ('fear3', 0), ('AffectDimension0', 0), ('fear0', 0), 
                         ('surprise3', 0), ('surprise1', 0), ('anger0', 0), 
                         ('anticipation1', 0), ('sadness3', 0), ('disgust3', 0), 
                         ('anger3', 0), ('joy3', 0), ('negative3', 0), 
                         ('anticipation3', 0)])
            for emotion, score in inner_dict.items():
                main_dict[word][emotion] = score
    
    return main_dict
    
    
def inport_sentiment_lexicons(meta_filepath):
    '''
    Takes file path with meta data of lexicons and their file paths.
    Reads the lexicons (with word, emotion and score)
    Creates a combined dictionary looking as following:
        {word1: {'positive3': '0', 'score2': '-0.577', 'fear3': '1', 
        'AffectDimension0': 0, 'fear0': '0.828', 'surprise3': '0', ...},
        word2: {'positive3': '1', 'score2': '0.4', 'fear3': '1', 
        'AffectDimension0': 1, 'fear0': '0.3', 'surprise3': '0', ...},...}
    Returns this dictionary of dictionaries.        
    '''
    meta_data_list = read_meta_file(filepath)
    list_of_lexicon_dicts = []
    for lexicon_meta_data in meta_data_list:
        lexicon_dict = term_score_emotion_to_dict(lexicon_meta_data)
        list_of_lexicon_dicts.append(lexicon_dict)
    
    main_dict = combine_dicts(list_of_lexicon_dicts)
    return main_dict
        
    
    