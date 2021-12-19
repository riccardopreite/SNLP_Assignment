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
                    help='model to test. "full" test on corpus_pos.txt, "slice" test on 50 (default) sentence of corpus_pos.txt, "example" test on corpus_example.txt, "obama" test on corpus_obama. Default slice')
parser.add_argument('--ite', type=int, nargs='?',
                    help='Number of itearation for the training. Default 20')
parser.add_argument('--size', type=int, nargs='?',
                    help='Train size. Default 50')

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

def obama(line: int, learning_rate: float, iteration_number:int, size:int):
    print("Training on obama corpus")
    corpus = import_corpus('corpus_obama.txt')

    model = LinearChainCRF()
    model.initialize([corpus[1]])

    model.train(iteration_number, learning_rate)
    save("obama", learning_rate, model,1)

    print("\n", model.most_likely_label_sequence(corpus[0]))
    print("\n",corpus[0])

def example(line: int, learning_rate: float, iteration_number:int, size:int):
    print("Training on example corpus")
    corpus = import_corpus('corpus_example.txt')

    model = LinearChainCRF()
    model.initialize(corpus[:2],2)

    model.train(iteration_number, learning_rate)
    save("example", learning_rate, model)

    print("\n", model.most_likely_label_sequence(corpus[2]))
    print("\n",corpus[2])

def slice(line: int, learning_rate: float, iteration_number:int, size: int):
    print("Training on slice corpus of size",size)
    corpus_full = import_corpus('corpus_pos.txt')
    corpus = []
    i = 0
    while len(corpus) < size:
        if i != line:   
            corpus.append(corpus_full[i])
        i+=1
    
    model = LinearChainCRF()
    model.initialize(corpus)

    model.train(iteration_number, learning_rate)
    save("slice", learning_rate, model, len(corpus))
    print("\n", model.most_likely_label_sequence(corpus[line]))
    print("\n",corpus[line])

def full(line: int, learning_rate: float, iteration_number:int, size: int):
    
    corpus_full = import_corpus('corpus_pos.txt')
    test_sentence = corpus_full.pop(line)
    print("Training on full corpus of size ",corpus_full)
    
    model = LinearChainCRF()
    model.initialize(corpus_full)

    model.train(iteration_number, learning_rate)
    save("full", learning_rate, model, len(corpus_full))
    print("\n", model.most_likely_label_sequence(test_sentence))
    print("\n",test_sentence)

def save(mode:str, learning_rate: float, model, size: int):
        exist = os.path.exists("models")
        if not exist:
            os.makedirs("models")

        date:str = datetime.datetime.now().strftime("day_%Y-%m-%d_time_%H-%M-%S_")
        name = "models/"+mode+"_"+date+'_lr_'+str(learning_rate)+"_train_size_"+str(size)+".pkl"
        pickle.dump(model, open( name, "wb" ),-1)

def load(filename: str) -> LinearChainCRF:
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        return model

def test(model: LinearChainCRF, test_line: int):
    corpus = import_corpus('corpus_pos.txt')
    print("\n", model.most_likely_label_sequence(corpus[test_line]))
    print("\n",corpus[test_line])

function = {"full":full, "slice":slice, "example":example, "obama":obama}
if __name__ == "__main__":
    args = parser.parse_args()

    line = args.line if args.line != None else 0
    learning_rate = args.lr if args.lr != None else 0.01
    mode = args.mode if args.mode != None else "slice"
    iteration_number = args.ite if args.ite != None else 20
    size = args.size if args.size != None else 50
    
    if args.test != None:
        '''Test'''
        file_name = args.test

        model = load(file_name)
        test(model, line)
    else:
        '''Train'''
        function[mode](line, learning_rate, iteration_number, size)
    
    

    
