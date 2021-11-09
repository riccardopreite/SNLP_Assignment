'''
Implement the matrix of emmision probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state:str, emission_symbol:str, internal_representation:dict)->float:
    label_key = state+"->"+emission_symbol
    if label_key in internal_representation.keys():
        print("Returning probabilities of emission: "+label_key)
        return internal_representation[label_key]
    print("ERROR - Unknown emission state tag:"+label_key+". Returning 0")
    return 0.0
    
    
'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus:list,unique_label:list,unique_token:list)->dict:
    emission_probabilities_dict:dict = {}
    for label in unique_label:
        for token in unique_token:
            label_key:str = label+"->"+token
            emission_probabilities_dict[label_key] = 0
        
    for sentence in corpus:
        for word in sentence:
            label_key:str = word[1]+"->"+word[0]
            if label_key in emission_probabilities_dict:
                emission_probabilities_dict[label_key] = emission_probabilities_dict[label_key] + 1
    total_occourence: int = sum(emission_probabilities_dict.values())
    emission_probabilities_dict = {label_transition: counter/total_occourence for label_transition,counter in emission_probabilities_dict.items() }

    return emission_probabilities_dict