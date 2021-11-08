from collections import Counter


'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parameterization of this probability distribuion;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state:str, internal_representation:dict) -> float:
    if state in internal_representation.keys():
        return internal_representation[state]
    print("ERROR - Unknown initial state tag. Returning 0")
    return 0.0

'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus) -> dict:
    first_word: list = []
    for sentence in corpus:
        first_word.append(sentence[0])
    
    counted_label : dict = Counter(word[1] for word in first_word)


    
    #total_occourence: int = sum(counted_label.values())
    initial_probabilities_dict: dict = {label: counter/sum(counted_label.values()) for label,counter in counted_label.items() } 
    # print(initial_probabilities_dict)
    return initial_probabilities_dict