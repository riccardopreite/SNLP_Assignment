################################################################################
## SNLP exercise sheet 4
################################################################################
from LinearChainCRF import LinearChainCRF


'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    
    
    with open(path_to_file) as f:
        for line in f:
            line = line.strip()
            
            if len(line) == 0:
                sentences.append(sentence)    
                sentence = []
                continue
                    
            pair = line.split(' ')
            sentence.append((pair[0], pair[-1]))
            
        if len(sentence) > 0:
            sentences.append(sentence)
                
    return sentences



if __name__ == "__main__":
    corpus = import_corpus('corpus_pos.txt')
    corpus = corpus[1:50]

    model = LinearChainCRF()
    model.initialize(corpus)

    model.train(20)
    print("\n", model.most_likely_label_sequence(corpus[0]))
    print("\n",corpus[0])

    
