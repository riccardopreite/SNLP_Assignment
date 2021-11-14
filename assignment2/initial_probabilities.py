# Riccardo Preite 4196104

from dict_utils import save_file

'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parameterization of this probability distribuion;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state:str, internal_representation:dict) -> float:
    """Check if the state exist in initial state and return it, else return 0.0"""
    if state in internal_representation.keys():
        # print("Returning probabilities of initial state: "+state)
        return internal_representation[state]
    # print("ERROR - Unknown initial state: "+state+" tag. Returning 0")
    return 0.0

'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus:list) -> dict:
    initial_probabilities_dict = dict()
    """First we count the occourence of each tag as init tag"""
    for sentence in corpus:
        init_tag = sentence[0][1]
        initial_probabilities_dict[init_tag] = initial_probabilities_dict.get(init_tag,0) + 1

    """Here we divide every counted init occourence for the amount of sentence in the corpus"""
    number_of_sentences = len(corpus)
    for label in initial_probabilities_dict:
        initial_probabilities_dict[label] =  initial_probabilities_dict[label] / number_of_sentences

    """Save json for controlling purpose"""
    json_path = "initial/initial_probabilities_dict.json"
    save_file(json_path,initial_probabilities_dict)
    
    return initial_probabilities_dict