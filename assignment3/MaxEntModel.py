import math
import random
from typing import Tuple
from create_feature import create_feature
import numpy as np


class MaxEntModel(object):
    # training corpus
    corpus: list = None

    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta: np.ndarray = None

    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    feature_indices: dict = None

    # set containing a list of possible lables
    # has to be set by the method 'initialize'
    labels: list = None

    active_features: dict = None
    empirical_feature_counts: dict = None
    train_count: int = None
    train_batch_count: int = None


    # Exercise 1 a) ###################################################################

    def initialize(self, corpus: list):
        '''
        Initialize the maximun entropy model, i.e., build the set of all features, the set of all labels
        and create an initial array 'theta' for the parameters of the model.
        Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
        '''
        self.active_features = dict()
        self.empirical_feature_counts = dict()
        self.train_count = 0
        self.train_batch_count = 0
        self.corpus = corpus
        self.feature_indices, self.labels = create_feature(corpus)
        self.theta = np.ones(max(self.feature_indices.values())+1)       
        # your code here

    # Exercise 1 b) ###################################################################

    def get_active_features(self, word: str, label: str, prev_label: str) -> np.ndarray:
        '''
        Compute the vector of active features.
        Parameters: word: string; a word at some position i of a given sentence
                    label: string; a label assigned to the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing only zeros and ones.
        '''

        active_features_key = (word, label, prev_label)
        if self.active_features.get(active_features_key, None) is not None:
            return self.active_features[active_features_key]
        
        keys_list = filter(
            lambda key_pair:
                (key_pair[0] == word and key_pair[1] == label)
                or 
                (key_pair[0] == prev_label and key_pair[1] == label),
            self.feature_indices.keys())
        theta_copy = np.zeros(len(self.theta))
        for key in keys_list:
            feature_index = self.feature_indices[key] 
            theta_copy[feature_index] = 1

        self.active_features[active_features_key] = np.array(theta_copy)
        
    
        return theta_copy


    # Exercise 2 a) ###################################################################

    def cond_normalization_factor(self, word: str, prev_label: str) -> float:
        '''
        Compute the normalization factor 1/Z(x_i).
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''

        '''
            For each index of active features with the given pair calculating the exponential
            e**(theta[i]*f(word,tag))  // f(word,tag) should be 1 for the way we defined the fun
            in the end sum overall e**(theta[i]*f(word,tag)) for each tag
         '''
        z = 0

        for tag in self.labels:
            multiplied_vector = self.theta * self.get_active_features(word, tag, prev_label)
            z += np.exp(sum(
                multiplied_vector[multiplied_vector > 0]
                )
            )

        
        return z

    # Exercise 2 b) ###################################################################

    def conditional_probability(self, label: str, word: str, prev_label: str) -> float:
        '''
        Compute the conditional probability of a label given a word x_i.
        Parameters: label: string; we are interested in the conditional probability of this label
                    word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''

        Z = self.cond_normalization_factor(word, prev_label)
        multiplied_vector = self.theta * self.get_active_features(word, label, prev_label)

        conditional_probability = (1/Z) * np.exp(
            sum(
                multiplied_vector[multiplied_vector > 0]
            )
        )

        return conditional_probability


    # Exercise 3 a) ###################################################################
    def empirical_feature_count(self, word: str, label: str, prev_label: str) -> np.ndarray:
        '''
        Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
        Parameters: word: string; a word x_i some position i of a given sentence
                    label: string; the actual label of the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the empirical feature count
        '''
        
        empirical_key = (word, label, prev_label)
        if self.empirical_feature_counts.get(empirical_key, None) is not None:
            return self.empirical_feature_counts[empirical_key]


        empirical_feature = np.zeros(len(self.theta))
        feature = self.get_active_features(word, label, prev_label)
        active_feature = np.where(feature>0)

        for index in active_feature:
            empirical_feature[index] = 1

        self.empirical_feature_counts[empirical_key] = empirical_feature
        return self.empirical_feature_counts[empirical_key]

    # Exercise 3 b) ###################################################################

    def expected_feature_count(self, word: str, prev_label: str) -> np.ndarray:
        '''
        Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
        (see variable theta)
        Parameters: word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the expected feature count
        '''
        expted_feature_count = np.zeros(len(self.theta))

        for label in self.labels:
            feature = self.get_active_features(label=label,word=word,prev_label=prev_label)
            probablity = self.conditional_probability(label=label,word=word,prev_label=prev_label)
            
            active_feature = np.where(feature>0)

            for index in active_feature:
                expted_feature_count[index] += probablity

        return expted_feature_count

    # Exercise 4 a) ###################################################################

    def parameter_update(self, word: str, label: str, prev_label: str, learning_rate: float):
        '''
        Do one learning step.
        Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                    label: string; the actual label of the selected word
                    prev_label: string; the label of the word at position i-1
                    learning_rate: float
        '''
        gradient: np.ndarray = self.empirical_feature_count(word,label,prev_label) - self.expected_feature_count(word,prev_label)
        
        self.theta = self.theta + (learning_rate * gradient)
        

    # Exercise 4 b) ###################################################################

    def train(self, number_iterations: int, learning_rate:int =0.1):
        '''
        Implement the training procedure.
        Parameters: number_iterations: int; number of parameter updates to do
                    learning_rate: float
        '''
        for iteration in range(number_iterations):
            random_sentence: int = random.choice(self.corpus)
            random_pair_index: int = random.randrange(len(random_sentence))


            word: str = random_sentence[random_pair_index][0]
            label: str = random_sentence[random_pair_index][1]
            prev_label: str = 'start' if random_pair_index == 0  else random_sentence[random_pair_index-1][1]
            
            self.train_count += 1
            self.parameter_update(word, label, prev_label, learning_rate)
        print(self.theta[self.theta != 1])

        # your code here

        pass

    # Exercise 4 c) ###################################################################

    def predict(self, word: str, prev_label: str) -> str:
        '''
        Predict the most probable label of the word referenced by 'word'
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: string; most probable label
        '''
        initial_label_probabilities: np.ndarray = np.zeros(len(self.labels))
        
        for i in range(len(self.labels)):
            initial_label_probabilities[i] = self.conditional_probability(word, self.labels[i], prev_label)

        return self.labels[np.argmax(initial_label_probabilities)]

    # Exercise 5 a) ###################################################################

    def empirical_feature_count_batch(self, sentences: list) -> np.ndarray:
        '''
        Predict the empirical feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the empirical feature count
        '''
        initial_empirical_feature_probabilities: np.ndarray = np.zeros(len(self.feature_indices))
        
        for sentence in sentences:
            index: int
            word_tag: Tuple
            for index, word_tag in enumerate(sentence):
                word = word_tag[0]
                label = word_tag[1]
                prev_label = 'start' if index == 0 else sentence[index-1][1]

                initial_empirical_feature_probabilities += self.empirical_feature_count(word,label,prev_label)
        return initial_empirical_feature_probabilities


    # Exercise 5 a) ###################################################################
    def expected_feature_count_batch(self, sentences: list):
        '''
        Predict the expected feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the expected feature count
        '''

        initial_empirical_feature_probabilities: np.ndarray = np.zeros(len(self.feature_indices))
        
        for sentence in sentences:
            index: int
            word_tag: Tuple
            for index, word_tag in enumerate(sentence):
                word = word_tag[0]
                label = word_tag[1]
                prev_label = 'start' if index == 0 else sentence[index-1][1]

                initial_empirical_feature_probabilities += self.expected_feature_count(word,label,prev_label)

        return initial_empirical_feature_probabilities

    # Exercise 5 b) ###################################################################
    def train_batch(self, number_iterations: int, batch_size: int, learning_rate: int=0.1):
        '''
        Implement the training procedure which uses 'batch_size' sentences from to training corpus
        to compute the gradient.
        Parameters: number_iterations: int; number of parameter updates to do
                    batch_size: int; number of sentences to use in each iteration
                    learning_rate: float
        '''

        self.theta = np.ones(len(self.theta))

        for iteration in range(number_iterations):
            sentences = random.sample(self.corpus,k=batch_size)
            for sentence in sentences:
                self.train_batch_count += len(sentence)
            self.theta = self.theta + learning_rate * (self.empirical_feature_count_batch(sentences) - self.expected_feature_count_batch(sentences))

        print(self.theta[self.theta != 1])
        
        # your code here

        pass
