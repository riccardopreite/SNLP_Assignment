'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state:str, to_state:str, internal_representation:dict)-> float:
    label_key = from_state+"->"+to_state
    if label_key in internal_representation.keys():
        print("Returning probabilities of transition: "+label_key)
        return internal_representation[label_key]
    print("ERROR - Unknown transition state tag:"+label_key+". Returning 0")
    return 0.0


'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus:list,unique_label:list)-> dict:
    transition_probabilities_dict:dict = {}
    for label in unique_label:
        for sentence in corpus:
            for word_index, word in enumerate(sentence):

                if word[1] == label and word_index+1 < len(sentence):
                    next_word = sentence[word_index+1]
                    label_key:str = word[1]+"->"+next_word[1]
                    if label_key in transition_probabilities_dict:
                        transition_probabilities_dict[label_key] = transition_probabilities_dict[label_key] + 1
                    else:
                        transition_probabilities_dict[label_key] = 1

    total_occourence: int = sum(transition_probabilities_dict.values())
    transition_probabilities_dict = {label_transition: counter/total_occourence for label_transition,counter in transition_probabilities_dict.items() }
    return transition_probabilities_dict