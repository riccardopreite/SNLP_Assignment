################################################################################
## SNLP exercise sheet 4
################################################################################
from LinearChainCRF import LinearChainCRF
# from new import LinearChainCRF


'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    f = open(path_to_file)

    while True:
        line = f.readline()
        if not line: break

        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)
            sentence = []
            continue

        parts = line.split(' ')
        sentence.append((parts[0], parts[-1]))

    f.close()
    return sentences



if __name__ == "__main__":
    corpus_full = import_corpus('corpus_pos.txt')
    corpus = corpus_full[1:20]

    model = LinearChainCRF()
    model.initialize(corpus)

    model.train(20)
    print("\n", model.most_likely_label_sequence(corpus[0]))
    print("\n",corpus[0])

    
