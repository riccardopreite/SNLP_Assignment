################################################################################
## SNLP exercise sheet 2
################################################################################
from collections import Counter
import sys
from inizial_probabilities import *
from transition_probabilities import *
from emission_probabilities import *

from read_corpus_file import read_corpus_file


CORPUS_FILE_NAME = "corpus_ner.txt"
  
    
def main():
    sentences = read_corpus_file(CORPUS_FILE_NAME)
    initial_state_probabilities_dict = estimate_initial_state_probabilities(sentences) 
    print(initial_state_probabilities('O',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-PER',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-ORG',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-LOC',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-MISC',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-NOT',initial_state_probabilities_dict))    
    
if __name__ == "__main__":
    main()



    
    
    
# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_smbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_smbols, initial_state_probabilities_parameters, transition_probabilities_parameters, emission_probabilities_parameters):
    pass















