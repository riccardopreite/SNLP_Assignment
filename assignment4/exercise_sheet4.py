################################################################################
## SNLP exercise sheet 4
################################################################################
from LinearChainCRF import LinearChainCRF
import argparse, datetime, os, random, pickle
# Instantiate the parser
parser = argparse.ArgumentParser(description='LinearChainCRF')
parser.add_argument('--line', type=int, nargs='?',
                    help='Index of line to observe > 0. Default 0')
parser.add_argument('--lr', type=float, nargs='?',
                    help='learning rate. Default=0.01')
parser.add_argument('--test', type=str, nargs='?',
                    help='file of the model to load')
parser.add_argument('--mode', type=str, nargs='?',
                    help='model to test. "full" test on corpus_pos.txt, "example" test on corpus_example.txt, "obama" test on corpus_obama. Default full')

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

def obama(line: int, learning_rate: float):
    print("Training on obama corpus")
    corpus = import_corpus('corpus_obama.txt')

    model = LinearChainCRF()
    model.initialize([corpus[1]])

    model.train(20, learning_rate)
    save("obama", learning_rate, model)

    print("\n", model.most_likely_label_sequence(corpus[0]))
    print("\n",corpus[0])

def example(line: int, learning_rate: float):
    print("Training on example corpus")
    corpus = import_corpus('corpus_example.txt')

    model = LinearChainCRF()
    model.initialize(corpus[:2])

    model.train(20, learning_rate)
    save("example", learning_rate, model)

    print("\n", model.most_likely_label_sequence(corpus[2]))
    print("\n",corpus[2])

def full(line: int, learning_rate: float):
    print("Training on full corpus")
    corpus_full = import_corpus('corpus_pos.txt')
    corpus = []
    if line != 0:
        for i in range(len(corpus)):
            if i != line:   
                corpus.append(corpus_full[i])
    else:
        corpus = corpus_full[1:50]
    
    model = LinearChainCRF()
    model.initialize(corpus)

    model.train(20, learning_rate)
    save("full", learning_rate, model)
    print("\n", model.most_likely_label_sequence(corpus[line]))
    print("\n",corpus[line])

def save(mode:str, learning_rate: float, model):
        exist = os.path.exists("models")
        if not exist:
            os.makedirs("models")

        date:str = datetime.datetime.now().strftime("day_%Y-%m-%d_time_%H-%M-%S_")
        name = "models/"+mode+"_"+date+'_lr'+str(learning_rate)+".pkl"
        print(model)
        pickle.dump(model, open( name, "wb" ),-1)

def load(filename: str) -> LinearChainCRF:
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        return model

def test(model: LinearChainCRF, test_line: int):
    corpus = import_corpus('corpus_pos.txt')
    print("\n", model.most_likely_label_sequence(corpus[test_line]))
    print("\n",corpus[test_line])

function = {"full":full,"example":example,"obama":obama}
if __name__ == "__main__":
    args = parser.parse_args()

    line = args.line if args.line != None else 0
    learning_rate = args.lr if args.lr != None else 0.01
    mode = args.mode if args.mode != None else "full"
    
    
    if args.test != None:
        '''Test'''
        file_name = args.test

        model = load(file_name)
        test(model, line)
    else:
        '''Train'''
        function[mode](line, learning_rate)
    
    

    
