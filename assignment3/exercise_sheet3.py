################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
from MaxEntModel import MaxEntModel

CORPUS_FILE_NAME = "corpus_pos.txt"

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''




# Exercise 5 c) ###################################################################
def evaluate(corpus):
    '''
    Compare the training methods 'train' and 'train_batch' in terms of convergence rate
    Parameters: corpus: list of list; a corpus returned by 'import_corpus'
    '''
    
    # your code here
    
    pass
    

def import_corpus(path_to_file: str) -> list:
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
            sentence.append((pair[0], pair[-1]))
            
        if len(sentence) > 0:
            sentences.append(sentence)
                
    return sentences


if __name__ == "__main__":
    corpus = import_corpus(CORPUS_FILE_NAME)

    maxEntropy: MaxEntModel = MaxEntModel()
    maxEntropy.initialize(corpus)
    print(maxEntropy.get_active_features('who','WP','NNS'))
    print(maxEntropy.cond_normalization_factor('who','NNS'))
    print(maxEntropy.conditional_probability('WP','who','NNS'))
    print(maxEntropy.empirical_feature_count('who','WP','NNS'))
    print(maxEntropy.expected_feature_count('who','NNS'))
    
    
    
    
    
    
    
