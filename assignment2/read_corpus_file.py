from collections import Counter
from typing import Union, Optional, List, Tuple, Dict

def read_corpus(CORPUS_FILE_NAME,unique_token:dict)-> list:
    sentences = []
    sentence = []
        
    with open(CORPUS_FILE_NAME) as f:
        for line in f:
            line = line.strip()
            
            if len(line) == 0:
                sentences.append(sentence)    
                sentence = []
                continue
                    
            pair = line.split(' ')
            token = lambda : '<unknown>' if pair[0].lower() in unique_token else pair[0].lower()
            sentence.append((token() , pair[-1]))
            
        if len(sentence) > 0:
            sentences.append(sentence)

    return sentences


def count_token(CORPUS_FILE_NAME) -> dict:
    with open(CORPUS_FILE_NAME) as f:
            token: list = []
            for line in f:
                line = line.strip()
                line_splitted = line.split(' ')
                token.append(line_splitted[0].lower())
            countered_token : dict = Counter(token)
            unique_token :dict = {token:counter for token,counter in countered_token.items() if counter == 1}
            return unique_token
    
def generate_unique_label_and_token(sentences:list)->Tuple[list,list]:
    label:list = []
    token_label:list = []
    for sentence in sentences:
            for word in sentence:
                if word[0] != '<unknown>':
                    label.append(word[1])
                    token_label.append(word[0])


    unique_label = list(set(label))
    unique_token = list(set(token_label))
    return unique_label,unique_token

def read_corpus_file(CORPUS_FILE_NAME) -> Tuple[list,list,list]:
    unique_token_once: dict = count_token(CORPUS_FILE_NAME)
    sentences: list = read_corpus(CORPUS_FILE_NAME,unique_token_once)
    unique_label,unique_token = generate_unique_label_and_token(sentences)
    
    return sentences,unique_label,unique_token


def main():
    unique_token: dict = count_token("corpus_ner.txt")
    sentences: list = read_corpus("corpus_ner.txt",unique_token)   
    print(sentences[-100:]) 
    
if __name__ == "__main__":
    main()