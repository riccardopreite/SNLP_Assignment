################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys

from numpy import random
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
    
    '''
    Create a test set T by randomly selecting 10% of all sentences from the provided corpus
    C. Use the set D = C 􀀀 T for training.
    2 Create two instances A and B of the class MaxEntModel. A will be trained by train and B
    by train_batch. Use the training corpus D for initialization.
    '''
    iteration: int = 200
    test_size: int = len(corpus) // 10
    test_corpus = random.sample(corpus,test_size)
    train_corpus = [sentence for sentence in corpus if sentence not in test_corpus]

    A = MaxEntModel()
    A.initialize(train_corpus)

    B = MaxEntModel()
    B.initialize(train_corpus)

    accuracies_A = list()
    word_numbers_A = list()
    accuracies_B = list()
    word_numbers_B = list()

    number_iteration_A: int = 1
    learning_rate_A: int = 1

    number_iteration_B: int = 0.1
    batch_size_B: int = 1
    learning_rate_B: int = 0.01

    for i in range(iteration):
        A.train(number_iteration_A, learning_rate_A)
        B.train_batch(number_iteration_B, batch_size_B, learning_rate_B)






    

    
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
    maxEntropy.get_active_features('who','WP','NNS')
    maxEntropy.cond_normalization_factor('who','NNS')
    maxEntropy.conditional_probability('WP','who','NNS')
    maxEntropy.empirical_feature_count('who','WP','NNS')
    maxEntropy.expected_feature_count('who','NNS')
    maxEntropy.train(10)