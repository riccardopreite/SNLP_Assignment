# Riccardo Preite 4196104

from dict_utils import save_file
'''
Implement the matrix of emmision probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state:str, emission_symbol:str, internal_representation:dict)->float:
    """Check if the state exist in initial state and return it, else return 0.0"""
    if state in internal_representation:

        return internal_representation[state].get(emission_symbol, 0.0)

    # print("ERROR - Unknown emission state tag:"+state+"->"+emission_symbol+". Returning 0.0")
    return 0.0
    
"""
    Function to count occourence of each label in corpus. 
    Checking the last (word,tag) because there is emission in the last word.
"""
def label_occourence(corpus:list)->dict:
    label_occourence_dict = dict()
    for sentence in corpus:
        for word in sentence:
            label_occourence_dict[word[1]] = label_occourence_dict.get(word[1], 0) + 1
    return label_occourence_dict
'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus:list)->dict:
    emission_probabilities_dict:dict = {}
    """Count occourence of each label in corpus"""
    label_occourence_dict = label_occourence(corpus)
    token_label_occourence:dict = {}

    """Count occourence of each word in corpus"""
    for sentence in corpus:
        for word in sentence:
            token_label_occourence[word] = token_label_occourence.get(word,0) + 1
    
    """Calculating probaility for each emission label->token"""
    for label in label_occourence_dict:
        sub_word_dict = {}
        
        """Get list of occourence for each label"""
        fit_first_label_list = list(filter(lambda token_label:token_label[1] == label,token_label_occourence))
        
        """Divide occourence of (token,label) by occourence of the label"""
        for word,label in fit_first_label_list:
            sub_word_dict[word] = token_label_occourence[(word,label)] / label_occourence_dict[label]
        
        """Append created dict of probability for label"""
        emission_probabilities_dict[label] = sub_word_dict

    """Save json for controlling purpose"""
    json_path = "emission/emission_probabilities_dict.json"
    save_file(json_path,emission_probabilities_dict)

    return emission_probabilities_dict