# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

# filepath_en = './lexicon/English/lexicon_meta_data_English.txt'
# filepath_ar = './lexicon/Arabic/lexicon_meta_data_Arabic.txt'

def read_meta_file(filepath):
    '''
    Reads tsv file containing meta information of lexica. 
    Returns list of lists containing the meta information. 
    '''
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
    '''
    Get the term/word, score and emotion from a tsv file using a meta_data_list.
    See read me for the specifications of the meta data file from which the 
    meta_data_list is produced.
    
    :param list meta_data_list: list of lists, see read me for more info
    
    Returns a dictionary where the key is a word and the value is a dictionary 
    matching emotions to their scores.
    '''    
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
            if any([split_line[0] == '\n', 
                    split_line[0] == 'term',
                    split_line[0] == '[Arabic Term]',
                    split_line[0] == '[English Term]',
                    split_line[0] == '\ufeff[English Term]']):
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
            if emotion == '[Emotion]3':
                print(split_line)
            # {word1: {emotion1:score, emotion2:score}, word2: {emotion1:score}}
            if word in word_dict:
                word_dict[word][emotion] = score
            else:
                word_dict[word] = dict()
                word_dict[word][emotion] = score
    return word_dict
 
def combine_dicts(list_of_dicts, emotion_set):
    '''
    Combines dictionaries from list of dictionaries to one larger dictionary.
    The keys of the dictionaries need to be of the same type (ie words of a lexicon).
    The values are inner dictionaries of which the keys are in the emotion set.
    
    :param list_of_dicts list: list of dictionaries which have inner dictionaries
        as keys
    :param emotion_set set: set of strings which are the keys of the inner dictionaries
    '''
    # Create main dictionary
    main_dict = dict()
    # Loop through all dictionaries in list of dictionaries
    for lexicon_dict in list_of_dicts:
        # Loop through all words in lexicon
        for word, inner_dict in lexicon_dict.items():
            # If the word is not not yet in the main dictionary
            # Then create an inner dictionary for it from the emotion_set 
            # And set all scores to 0
            if word not in main_dict:
                main_dict[word] = {emotion:0 for emotion in emotion_set}
            # For every emotion in the inner dict of the lexicon
            # And the score to the inner dictionary of the word of the main dictionary
            for emotion, score in inner_dict.items():
                main_dict[word][emotion] = score

    return main_dict
    
    
def import_sentiment_lexicons(meta_filepath):
    '''
    Takes file path with meta data of lexicons (file paths and column indices 
    containing relevant information).
    Reads all the lexicons (with word, emotion and score), and stored them in 
    dictionaries. All these individual lexicon dictionaries are stored in a list.
    Finds set of all emotions from all lexicons.
    Creates a combined dictionary looking as following:
        {word1: {'positive': '0', 'fear': 0.453, ...},
         word2: {'positive': '1', 'fear': '1'...},
         ...}
        
    :param meta_filepath: filepath the to meta file. Readme had more information
        on structure of this file.    
    
    Returns this dictionary of dictionaries.        
    '''
    # Get list of lists containing meta data (filepath to lexicon and indices
    # of relevant columns)
    meta_data_list = read_meta_file(meta_filepath)
    
    # Create list of dictionaries containing all the lexicons
    list_of_lexicon_dicts = []
    # For each lexicon create a dictionary of the word and the emotion scores
    # and append this dictionary to the list of lexicon dictionaries
    for lexicon_meta_data in meta_data_list:
        lexicon_dict = term_score_emotion_to_dict(lexicon_meta_data)
        list_of_lexicon_dicts.append(lexicon_dict)    
    
    # Create a set of the emotion labels
    emotion_set = set()
    # For each lexicon
    for lexicon_dict in list_of_lexicon_dicts:
        # For each dictionary corresponding to a word in the lexicon
        for word, inner_dict in lexicon_dict.items():
            for emotion, score in inner_dict.items():
                # For each emotion add it to the set of all emotions
                emotion_set.add(emotion)
    
    # Combine the individual dictionaries of words and the emotion scores (with
    # various emotions as stored in the emotion set) to one large dictionary
    main_dict = combine_dicts(list_of_lexicon_dicts, emotion_set)
    
    return main_dict
