from collections import Counter
from typing import Union, Optional, List, Tuple, Dict

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the second layer list contains tuples (token,label);

Every word is trasformed in lower case
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    
    
    with open(path_to_file) as f:
        for line in f:
            line = line.strip()
            
            if len(line) == 0:
                sentences.append(sentence)    
                sentence = []
                continue
                    
            pair = line.split(' ')
            sentence.append((pair[0].lower(), pair[-1]))
            
        if len(sentence) > 0:
            sentences.append(sentence)
                
    return sentences

"""Count token that occoure only once"""
def count_token(corpus:list) -> dict:
    token: list = []
    for sentence in corpus:
        for word in sentence:
            token.append(word[0])

    countered_token : dict = Counter(token)
    unique_token :dict = {token:counter for token,counter in countered_token.items() if counter == 1}
    return unique_token
    
"""Split corpus and observed_symobl"""
def split_corpus(sentences:list,index_observed_symbol:int) -> Tuple[list,list,list]:
    if index_observed_symbol >= len(sentences):
        raise IndexError("Index passed as argument is higher than corpus len")
    if index_observed_symbol < 0 :
        raise IndexError("Index passed as argument is lower than 0")
    corpus = sentences[0:index_observed_symbol] + sentences[index_observed_symbol:]
    observed_symbol = []
    observed_symbol_label = []
    for word in sentences[index_observed_symbol]:
        observed_symbol.append(word[0])
        observed_symbol_label.append(word[1])
    return corpus,observed_symbol,observed_symbol_label


"""Replace token that occoure just once with <unknown>"""
def unknown_token(corpus:list,observed_simbols:list)->list:
    unique_token = count_token(corpus)
    for sentence in corpus:
        for word_token_index in range(len(sentence)):
            word_token = sentence[word_token_index]
            word = word_token[0]
            label = word_token[1]
            if word in unique_token:
                sentence[word_token_index] = ("<unknown>",label)
    
    for word_index in range(len(observed_simbols)):
        if observed_simbols[word_index] in unique_token:
            observed_simbols[word_index] = "<unknown>"

    return corpus,observed_simbols

"""Function to calculate the unique token and read the corpus file"""
def read_corpus_file(CORPUS_FILE_NAME:str,index_observed_symbol) -> Tuple[list,list]:
    # unique_token_once: dict = count_token(CORPUS_FILE_NAME)
    
    sentences: list = import_corpus(CORPUS_FILE_NAME)
    sentences,observed_symbol,observed_symbol_label = split_corpus(sentences,index_observed_symbol)

    sentences,observed_symbol = unknown_token(sentences,observed_symbol)
    return sentences, observed_symbol,observed_symbol_label

def main():
    unique_token: dict = count_token("corpus_ner.txt")
    sentences: list = import_corpus("corpus_ner.txt",unique_token)   
    print(sentences[-100:]) 
    
if __name__ == "__main__":
    main()