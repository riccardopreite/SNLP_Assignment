################################################################################
## SNLP exercise sheet 3
################################################################################
import matplotlib.pyplot as plt
from typing import Tuple
import random
from MaxEntModel import MaxEntModel

CORPUS_FILE_NAME = "corpus_pos.txt"

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''




def test_model(A: MaxEntModel, B: MaxEntModel, test_corpus: list) ->  Tuple[float,float]:
    correct_A = 0
    correct_B = 0
    total_words = 0
    print('Start testing')
    for sentence in test_corpus:
        total_words += len(sentence)

        for i in range(len(sentence)):
            word = sentence[i][0]
            label = sentence[i][1]

            prevLabel = "start" if i == 0 else sentence[i-1][1]
            predicted_for_A = A.predict(word, prevLabel)
            predicted_for_B = B.predict(word, prevLabel)
            correct_A += 1 if predicted_for_A == label else 0
            correct_B += 1 if predicted_for_B == label else 0
    accuracy_A: float  = correct_A / total_words
    accuracy_B: float = correct_B / total_words
    print("End testing")
    print ("Total test words: ", total_words)
    print(correct_A)
    print(correct_B)

    return accuracy_A,accuracy_B


def plot_accuracy(color: str, word_number: list, accuracy: list, plot_name: str):
        plt.plot(word_number, accuracy, color=color)

        x_axis_max = max(word_number) + 25
        x_axis_min = min(word_number) - 25
        plt.xlabel("Number of training words")
        plt.xlim([x_axis_min,x_axis_max])

        plt.ylabel("Accuracy")
        plt.ylim([0,1.0])
        
        plt.title(plot_name)
        plt.savefig(plot_name+'.png')
        #plt.show()
        plt.clf()

# Exercise 5 c) ###################################################################
def evaluate(corpus):
    '''
    Compare the training methods 'train' and 'train_batch' in terms of convergence rate
    Parameters: corpus: list of list; a corpus returned by 'import_corpus'
    '''
    
    '''
    Create a test set T by randomly selecting 10% of all sentences from the provided corpus
    C. Use the set D = C 􀀀 T for training.
    2 Create two instances A and B of the class MaxEntModel. A will be trained by train and B
    by train_batch. Use the training corpus D for initialization.
    '''
    iteration: int = 200
    test_size: int = len(corpus) // 10
    test_corpus = random.sample(corpus,k=test_size)
    train_corpus = [sentence for sentence in corpus if sentence not in test_corpus]

    print(len(test_corpus))
    print(len(train_corpus))

    A = MaxEntModel()
    A.initialize(train_corpus)

    B = MaxEntModel()
    B.initialize(train_corpus)

    accuracies_A = list()
    word_numbers_A = list()
    accuracies_B = list()
    word_numbers_B = list()

    number_iteration_A: int = 1
    learning_rate_A: int = 1

    number_iteration_B: int = 1
    batch_size_B: int = 1
    learning_rate_B: int = 0.01
    print('Start training')
    for i in range(iteration):
        A.train(number_iteration_A, learning_rate_A)
        B.train_batch(number_iteration_B, batch_size_B, learning_rate_B)

        if i % 10 == 0:
            accuracy_A,accuracy_B = test_model(A,B,test_corpus)

            accuracies_A.append(accuracy_A)
            accuracies_B.append(accuracy_B)

            word_numbers_A.append(A.train_count)
            word_numbers_B.append(B.train_batch_count)
    
    print('End training')
    print (accuracies_A)
    print (accuracies_B)

    plot_accuracy(A, word_numbers_A, accuracies_A, 'Model A')
    plot_accuracy(B, word_numbers_B, accuracies_B, 'Model B')

def import_corpus(path_to_file: str) -> list:
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
    corpus = import_corpus(CORPUS_FILE_NAME)
    corpus_prova = corpus[:200]
    evaluate(corpus_prova)