################################################################################
## SNLP exercise sheet 2
################################################################################
from collections import Counter
import sys
from initial_probabilities import *
from transition_probabilities import *
from emission_probabilities import *
from viterbi import *
from read_corpus_file import read_corpus_file


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




def main():
    sentences = read_corpus_file(CORPUS_FILE_NAME)
    
    initial_state_probabilities_dict:dict = estimate_initial_state_probabilities(corpus=sentences) 
    # test_initial_state_probabilities(initial_state_probabilities_dict)
    transition_state_probabilities_dict:dict = estimate_transition_probabilities(corpus=sentences)
    # test_transition_state_probabilities(transition_state_probabilities_dict)
    emission_state_probabilities_dict:dict = estimate_emission_probabilities(corpus=sentences)
    # test_emission_state_probabilities(emission_state_probabilities_dict)

    wrong = 0
    wrong_sentence = []
    for sentence in sentences:
        sentence_tag = []
        sentence_viterbi = []
        for word in sentence:
            sentence_viterbi.append(word[0])
            sentence_tag.append(word[1])
        viterbi_tag = most_likely_state_sequence(sentence_viterbi, initial_state_probabilities_dict, transition_state_probabilities_dict, emission_state_probabilities_dict)
        if sentence_tag != viterbi_tag:
            sequence_dict = {
                "index": sentences.index(sentence),
                "generated":viterbi_tag,
                "original":sentence_tag
            }
            wrong_sentence.append(sequence_dict)
            wrong = wrong + 1
    
    json_path = "result/wrong_sequence_generated.json"
    save_file(json_path,wrong_sentence)
    print("Number of wrong tag sequence matched ",wrong , " on " , len(sentences))

    
if __name__ == "__main__":
    main()






