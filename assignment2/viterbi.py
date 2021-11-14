# Exercise 2 ###################################################################
from operator import itemgetter
from initial_probabilities import *
from transition_probabilities import *
from emission_probabilities import *
from math import inf,log2

"""Function to return -inf if the argument is 0"""

def infinite_log(to_log:float)->float:
    if not to_log:
        return -inf
    else:
        return log2(to_log)

''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_smbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_simbols:list, initial_state_probabilities_parameters:dict, transition_probabilities_parameters:dict, emission_probabilities_parameters:dict)-> list:
    initial_simbol = observed_simbols[0]
    label_set = list(emission_probabilities_parameters.keys())

    """Creating -inf viterbi matrix"""
    viterbi_matrix = [[-inf for symbol_col in range(len(observed_simbols))] for tag_row in range(len(label_set))]
    
    """Initialization"""
    current_max = -inf
    for init_label_index,init_label in enumerate(label_set):
        
        init_prob = initial_state_probabilities(init_label, initial_state_probabilities_parameters)
        emission_prob = emission_probabilities(init_label, initial_simbol, emission_probabilities_parameters)
        """Calculating initial symbol probabilities"""
        viterbi_matrix[init_label_index][0] = infinite_log(init_prob) + infinite_log(emission_prob)

    """Induction"""

    """arg_max is the list with every tuple of (label,probability) for each symbol"""
    arg_max = []

    """For each symbol after first"""
    for t in range(1,len(observed_simbols)):
        current_word = observed_simbols[t]
        arg_max.append([])
        """For each existing label"""
        for label_j_index,label_j in enumerate(label_set):
            current_max = -inf
            
            """Calculating with every existing label the transition and emission probability"""
            for label_i_index,label_i in enumerate(label_set):
                transition_probability_induction = transition_probabilities(label_i,label_j,transition_probabilities_parameters)
                emission_probability_induction = emission_probabilities(label_j,current_word,emission_probabilities_parameters)
                
                """Calculating probability with log2"""
                current_delta_probability = viterbi_matrix[label_i_index][t-1] + infinite_log(transition_probability_induction) + infinite_log(emission_probability_induction)
                
                """Updating current max for viterbi matrix"""
                if current_delta_probability > current_max:
                    current_max = current_delta_probability
                
                arg_max[t-1].append((label_i,current_delta_probability))

            """Add max prob to viterbi matrix"""
            viterbi_matrix[label_j_index][t] = current_max

    """Total"""
    tag_viterbi = []

    last_word_index = len(observed_simbols)-1
    tmp_max = -inf
    tmp_max_label = ""
    """Calculating last label by last symbol column in viterbi_matrix"""
    for label_index in range(0, len(label_set)):

        if viterbi_matrix[label_index][last_word_index] > tmp_max:
            tmp_max = viterbi_matrix[label_index][last_word_index]
            tmp_max_label = label_set[label_index]
            
    tag_viterbi.append(tmp_max_label)

    """Getting max probability for each symbol remaining from arg_max"""
    for t in range(len(arg_max)-1,-1,-1):
        tag_viterbi.append(max(arg_max[t],key=itemgetter(1))[0])
    
    """Reversing the list to get the right order"""
    tag_viterbi.reverse()
    return tag_viterbi
        