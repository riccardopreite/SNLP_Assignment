# Riccardo Preite 4196104
from random import random

"""Generate word sample function according to summation in the slides."""

def generate_word(words:list):
    random_x = random()
    summation = 0
    for i in range(0, len(words)):
        summation += words[i][1]
        if (summation - random_x >= 0):
            return words[i]


"""
This funcyion is used to filter the dict according to a filter. This is needed for bigram an trigram 
Because we have to choose word according to the previous or the two previous ones.
"""
def filt_list(filter_str:str,dict:dict,isBigram:bool):
    if isBigram:
        filtered = filter(lambda word: word[0][0] == filter_str, dict)
    else:
        filtered = filter(lambda word: (word[0][0],word[0][1]) == filter_str, dict)
    return list(filtered)

"""Sort dict to choose first the word with higher prob"""

def sortDict(dict:dict) -> list:
    return sorted(dict.items(),  key=lambda word: word[1], reverse=True)