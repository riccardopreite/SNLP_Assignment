################################################################################
## SNLP exercise sheet 3
# Riccardo Preite 4196104
################################################################################
import matplotlib.pyplot as plt
from typing import Tuple
import random
from MaxEntModel import MaxEntModel
import datetime
CORPUS_FILE_NAME = "corpus_pos.txt"

def test_model(A: MaxEntModel, B: MaxEntModel, test_corpus: list) ->  Tuple[float,float]:
    correct_A = 0
    correct_B = 0
    total_words = 0
    for sentence in test_corpus:
        total_words += len(sentence)

        for i in range(len(sentence)):
            word = sentence[i][0]
            label = sentence[i][1]
            prevLabel = "start" if i == 0 else sentence[i-1][1]
            predicted_for_A = A.predict(word, prevLabel)
            predicted_for_B = B.predict(word, prevLabel)
            if predicted_for_A == label:
                correct_A += 1 #if predicted_for_A == label else 0
            if predicted_for_B == label:
                correct_B += 1 #if predicted_for_B == label else 0

    accuracy_A: float  = correct_A / total_words
    accuracy_B: float = correct_B / total_words
    
    print ("\t\tTotal test words: ", total_words)
    print("\t\tAccuracy_A: ", accuracy_A, "Correct A: ", correct_A)
    print("\t\tAccuracy_B: ", accuracy_B, "Correct B: ", correct_B)
    return accuracy_A,accuracy_B


def plot_accuracy(color: str, word_number: list, accuracy: list, plot_name: str, learning_rate: int, train_test_size: int, date: str):
        plt.plot(word_number, accuracy, color=color)

        x_axis_max = max(word_number) + 25
        x_axis_min = min(word_number) - 25
        plt.xlabel("Number of training words")
        plt.xlim([x_axis_min,x_axis_max])

        plt.ylabel("Accuracy")
        plt.ylim([0,1.0])
        
        plt.title(plot_name)
        plt.savefig(date+plot_name+'_lr'+str(learning_rate)+'_train'+str(train_test_size)+'.png')
        plt.show()
        plt.clf()

# Exercise 5 c) ###################################################################
def evaluate(corpus: list):
    '''
    Compare the training methods 'train' and 'train_batch' in terms of convergence rate
    Parameters: corpus: list of list; a corpus returned by 'import_corpus'
    '''
    iteration: int = 200
    test_size: int = len(corpus) // 10
    test_corpus: list = random.sample(corpus,k=test_size)
    train_corpus: list = [sentence for sentence in corpus if sentence not in test_corpus]
    
    print("Train size:",len(train_corpus))
    print("Test size:",len(test_corpus))

    A: MaxEntModel = MaxEntModel()
    print('Initializing A...')
    A.initialize(train_corpus)
    print('Finished initializing A...')

    B: MaxEntModel = MaxEntModel()
    print('Initializing B...')
    B.initialize(train_corpus)
    print('Finished initializing B...')

    accuracies_A = list()
    word_numbers_A = list()
    accuracies_B = list()
    word_numbers_B = list()

    number_iteration_A: int = 1
    learning_rate_A: float = 0.1

    number_iteration_B: int = 1
    batch_size_B: int = 1
    learning_rate_B: float = 0.01
    print('Printing configuration:')
    print('\tConfiguration A:')
    print('\t\tNumber of iteration: ',number_iteration_A)
    print('\t\tLearning Rate: ',learning_rate_A)
    print('\tConfiguration B:')
    print('\t\tNumber of iteration: ',number_iteration_B)
    print('\t\tLearning Rate: ',learning_rate_B)
    print('\t\tBatch Size: ',batch_size_B)
    print('Start training...')
    for i in range(iteration):
        A.train(number_iteration_A, learning_rate_A)
        B.train_batch(number_iteration_B, batch_size_B, learning_rate_B)

        if (i+1) % 10 == 0:
            print('\tStarting test number: {}...'.format(((i+1)/10)))
            accuracy_A, accuracy_B = test_model(A,B,test_corpus)
            print('\tEnded test number: {}...'.format(((i+1)/10)))

            accuracies_A.append(accuracy_A)
            accuracies_B.append(accuracy_B)

            word_numbers_A.append(A.word_counter_vector[i])
            word_numbers_B.append(B.word_batch_counter_vector[i])
    
    print('...End training. Printing accuracy...')
    print ('Accuracy A:\n',accuracies_A)
    print ('Accuracy B:\n',accuracies_B)
    date:str = datetime.datetime.now().strftime("day_%Y-%m-%d_time_%H-%M-%S_")
    plot_accuracy('blue', word_numbers_A, accuracies_A, 'a_model', learning_rate_A, len(train_corpus), date)
    plot_accuracy('green', word_numbers_B, accuracies_B, 'b_model', learning_rate_B, len(train_corpus), date)

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

