from dict_utils import save_file
'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state:str, to_state:str, internal_representation:dict)-> float:
    """Check if the state exist in initial state and return it, else return 0.0"""
    if from_state in internal_representation:

        return internal_representation[from_state].get(to_state, 0.0)

    # print("ERROR - Unknown transition state tag:"+from_state+"->"+to_state+". Returning 0.0")
    return 0.0
    

"""
    Function to count occourence of each label in corpus. 
    Not checking the last (word,tag) because there is no transition from last word to another one.
"""
def label_occourence(corpus:list)->dict:
    label_occourence_dict = dict()
    for sentence in corpus:
        for word in sentence[:-1]:
            label_occourence_dict[word[1]] = label_occourence_dict.get(word[1], 0) + 1
    return label_occourence_dict

'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus:list)-> dict:
    
    transition_probabilities_dict:dict = {}
    label_to_label_occurence:dict = {}

    """Count occourence of each label in corpus"""
    label_occourence_dict = label_occourence(corpus)
    
    """Count occourence of each pair of (label,label_next) in corpus"""
    for sentence in corpus:
        for word_index in range(0,len(sentence)):
            word = sentence[word_index]
            if word_index+1 < len(sentence):
                next_word = sentence[word_index+1]
                label_key:str = (word[1],next_word[1])
                label_to_label_occurence[label_key] = label_to_label_occurence.get(label_key,0)+ 1     

    """For each existing label in the corpus"""
    for label in label_occourence_dict:
        fit_first_label: dict = {}
        
        """For each pair of label counted count his probability by dividing for the occourence of the first label"""
        for transition_tuple in label_to_label_occurence:
            from_state:str = transition_tuple[0]
            to_state:str = transition_tuple[1]
            
            if from_state == label:
                fit_first_label[to_state] = label_to_label_occurence[transition_tuple] / label_occourence_dict[label]
        
        """Append created dict of probability for label"""
        transition_probabilities_dict[label] = fit_first_label

    """Save json for controlling purpose"""
    json_path = "transition/transition_probabilities_dict.json"
    save_file(json_path,transition_probabilities_dict)

    return transition_probabilities_dict