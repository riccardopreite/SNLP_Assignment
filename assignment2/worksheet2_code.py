# Riccardo Preite 4196104

################################################################################
## SNLP exercise sheet 2
################################################################################
from initial_probabilities import *
from transition_probabilities import *
from emission_probabilities import *
from viterbi import *
from read_corpus_file import read_corpus_file
import argparse


# Instantiate the parser
parser = argparse.ArgumentParser(description='Viterbi alg on corpus')
parser.add_argument('--line', type=int, nargs='?',
                    help='Index of line to observe > 0')

CORPUS_FILE_NAME = "corpus_ner.txt"
  
def test_initial_state_probabilities(initial_state_probabilities_dict:dict):
    print(initial_state_probabilities('O',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-PER',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-ORG',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-LOC',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-MISC',initial_state_probabilities_dict))    
    print(initial_state_probabilities('B-NOT',initial_state_probabilities_dict))    
    
def test_transition_state_probabilities(transition_state_probabilities_dict:dict):
    print(transition_probabilities('I-LOC','B-MISC',transition_state_probabilities_dict))    
    print(transition_probabilities('O','O',transition_state_probabilities_dict))    
    print(transition_probabilities('I','B-MISC',transition_state_probabilities_dict)) 

def test_emission_state_probabilities(emission_state_probabilities_dict:dict):
    print(emission_probabilities('B-LOC','bailey',emission_state_probabilities_dict))    
    print(emission_probabilities('B-LOC','slovakia',emission_state_probabilities_dict))    
    print(emission_probabilities('B-LOC','<unknown>',emission_state_probabilities_dict)) 


def calculate_viterbi(index_observe:int):
    
    sentences,observed_symbol,observed_symbol_label = read_corpus_file(CORPUS_FILE_NAME,index_observe)
    
    initial_state_probabilities_dict:dict = estimate_initial_state_probabilities(corpus=sentences) 
    # test_initial_state_probabilities(initial_state_probabilities_dict)
    transition_state_probabilities_dict:dict = estimate_transition_probabilities(corpus=sentences)
    # test_transition_state_probabilities(transition_state_probabilities_dict)
    emission_state_probabilities_dict:dict = estimate_emission_probabilities(corpus=sentences)
    # test_emission_state_probabilities(emission_state_probabilities_dict)
    viterbi_tag = most_likely_state_sequence(observed_symbol, initial_state_probabilities_dict, transition_state_probabilities_dict, emission_state_probabilities_dict)
    # print(viterbi_tag)
    print("SEQUENCE GENERATED:")
    print(viterbi_tag)

    print("ORIGINAL SEQUENCE:")
    print(observed_symbol_label)
    if observed_symbol_label != viterbi_tag:
        print("Sequence are different")
        sequence_err_dict = {
                "index": index_observe,
                "generated":viterbi_tag,
                "original":observed_symbol_label
        }
        return sequence_err_dict,
    return {}

def main():
    observed_index = 5
    args = parser.parse_args()
    if args.line != None:
        observed_index = args.line
    error = calculate_viterbi(observed_index)
    
    json_path = "result/wrong_sequence_generated.json"
    save_file(json_path,error)
    return

if __name__ == "__main__":
    main()






