# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

filepath_1 = "./En-sent-lex/NRC-Affect-Intensity-Lexicon/NRC-AffectIntensity-Lexicon.txt"
filepath_2 = './En-sent-lex/AutomaticallyGeneratedLexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt'
filepath_3 = './En-sent-lex/AutomaticallyGeneratedLexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt'
filepath_4 = './En-sent-lex/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

def read_term_score_emotion(filename, word_index=0, score_index=1, emotion_index=-1):
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
            # If emotion index is given, have the emotion be the key of the dictionary for each word
            # {word1: {emotion1:score}, word2: {emotion1:score}}
            if emotion_index != -1:
                emotion = split_line[emotion_index]
                if word in word_dict:
                    word_dict[word][emotion] = score
                else:
                    word_dict[word] = dict()
                    word_dict[word][emotion] = score
            # If there is no emotion specified, then do not include key=word and value=score
            # {word1: score, word2: score}
            if emotion_index == -1:
                word_dict[word] = score
    return word_dict
 
#print(read_term_score_emotion(filepath, word_index=0, score_index=1, emotion_index=2))
#print(read_term_score_emotion(filepath_2, word_index=1, score_index=2, emotion_index=0))
#print(read_term_score_emotion(filepath_3, word_index=0, score_index=1))
#print(read_term_score_emotion(filepath_4, word_index=0, score_index=2, emotion_index=1))   
'''
to do:
    write to jsons
    Figure out how to combine data
'''