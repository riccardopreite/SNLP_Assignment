import math
import sys
import random
import numpy as np
from typing import Tuple

def create_feature_indices_and_tags(corpus: list) -> Tuple[set, dict]:
            words_unique = set()
            tags_unique = set("start")
            feature_set = set()
            for sentence in corpus:
                for word_tag in sentence:
                    words_unique.add(word_tag[0])
                    tags_unique.add(word_tag[1])
            
            feature_set.update([(words, label) for words in words_unique for label in tags_unique])
            feature_set.update([(label1, label2) for label1 in tags_unique for label2 in tags_unique])
            
            feature_index = 0
            feature_indices: dict = dict()
            for feature in feature_set:
                feature_indices[feature] = feature_index
                feature_index += 1

            return tags_unique, feature_indices

class LinearChainCRF(object):
    # training corpus
    corpus: list = list()
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta: np.ndarray = np.array([])
    
    # set containing all features observed in the corpus 'self.corpus'
    # choose an appropriate data structure for representing features
    # each element of this set has to be assigned to exactly one component of the vector 'self.theta'
    features: set = set()
    current_forward_matrix: list = list()
    current_backward_matrix: list = list()
    # set containing all lables observed in the corpus 'self.corpus'
    labels: set = set()
   
    
    def initialize(self, corpus):
        '''
        build set two sets 'self.features' and 'self.labels'
        '''
        self.corpus = corpus

        self.labels = set()
        self.active_features = dict()
        self.empirical_features = dict()
        self.features = dict()

        
        self.labels, self.features = create_feature_indices_and_tags(self.corpus)    
        self.theta = np.ones(len(self.features))
        print('Model initialized with a theta of len:',len(self.theta))

    def get_active_features(self, word: str, label: str, prev_label: str) -> np.ndarray:
        '''
        Compute the vector of active features.
        Parameters: word: string; a word at some position i of a given sentence
                    label: string; a label assigned to the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing only zeros and ones.
        '''
        active_feature_key: tuple = (word, label, prev_label)
        if active_feature_key in self.active_features.keys():
            return self.active_features[active_feature_key]
        
        keys_list = list(
            filter(
                lambda key_pair:
                    (key_pair[0] == word and key_pair[1] == label)
                    or 
                    (key_pair[0] == prev_label and key_pair[1] == label),
                self.features.keys()
            )
        )
        active_features: list = list(set([self.features[key] for key in keys_list]))
        
        self.active_features[active_feature_key] = np.array(active_features)        
        return self.active_features[active_feature_key]


    def empirical_feature_count(self, word: str, label: str, prev_label: str) -> np.ndarray:
        '''
        Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
        Parameters: word: string; a word x_i some position i of a given sentence
                    label: string; the actual label of the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the empirical feature count
        '''
        
        empirical_feature_key: tuple = (word, label, prev_label)
        if empirical_feature_key in self.empirical_features.keys():
            return self.empirical_features[empirical_feature_key]
            
        empirical_theta: np.ndarray = np.zeros(len(self.features))
        active_feature:np.ndarray = self.get_active_features(word, label, prev_label)

        for active_feature_index in active_feature:
            empirical_theta[active_feature_index] = 1
            
        self.empirical_features[empirical_feature_key] = empirical_theta
        return self.empirical_features[empirical_feature_key]


    def psi(self, word: str, label: str, prev_label: str):
        activeFeatures = self.get_active_features(word, label, prev_label)
        summed = sum(map(lambda i: self.theta[i], activeFeatures))
        return np.exp(summed)

    # Exercise 1 a) ###################################################################
    def forward_variables(self, sentence: list) -> list:
        '''
        Compute the forward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of forward variables
        '''
        if len(self.current_forward_matrix) != 0:
            return self.current_forward_matrix

        forward_matrix: list = [dict() for pair in range(len(sentence))]
        '''Initialization'''
        word: str = sentence[0][0]
        prev_label: str = "start"

        for tag_index, tag in enumerate(self.labels):
            forward_matrix[0][tag] = self.psi(word, tag, prev_label) 
            
        '''Recursion'''
        for t in range(1,len(sentence)):
            
            word_t = sentence[t][0]
            for tag_index, tag_j in enumerate(self.labels):
                new_alpha: float = 0.0
                for tag_index_i, tag_i in enumerate(self.labels):
                    psi = self.psi(word, tag_j, tag_i)
                    prevAlpha = forward_matrix[t-1][tag_i]
                    new_alpha += psi * prevAlpha
                forward_matrix[t][tag_j] = new_alpha

        self.current_forward_matrix = forward_matrix
        return forward_matrix        
        
        
    def backward_variables(self, sentence) -> np.ndarray:
        '''
        Compute the backward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of backward variables
        '''
        if len(self.current_backward_matrix) != 0:
            return self.current_backward_matrix

        backward_matrix: list = [dict() for pair in range(len(sentence))]

        '''Initialization'''
        for tag_index, tag in enumerate(self.labels):
            backward_matrix[len(sentence)-1][tag] = 1
        
        '''Recursion'''
        for t in range(len(sentence)-2,-1,-1):
            
            word_t = sentence[t+1][0]
            for tag_index, tag_j in enumerate(self.labels):
                new_alpha: float = 0.0
                for tag_index_i, tag_i in enumerate(self.labels):
                    psi = self.psi(word_t, tag_i, tag_j)
                    alpha_prev = backward_matrix[t+1][tag_i]
                    new_alpha += psi * alpha_prev
                
                backward_matrix[t][tag_j] = new_alpha

        self.current_backward_matrix = backward_matrix
        return backward_matrix
        
        
        
    
    # Exercise 1 b) ###################################################################
    def compute_z(self, sentence) -> float:
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        '''
        
        forward_matrix: np.ndarray = self.forward_variables(sentence)
        Z: float = 0.0
        for label in self.labels:
            Z += forward_matrix[-1][label]

        return Z

        
        
            
    # Exercise 1 c) ###################################################################
    def marginal_probability(self, sentence: list, t:int, y_t: str, y_t_minus_one: str) -> float:
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
        Returns: float: probability;
        '''
        forward_matrix: np.ndarray = self.forward_variables(sentence)
        backward_matrix: np.ndarray = self.backward_variables(sentence)
        Z: float = self.compute_z(sentence)
        word = sentence[t][0]

        psi = self.psi(word, y_t, y_t_minus_one)
        alpha = 1 if t == 0 else forward_matrix[t-1][y_t_minus_one]
        beta = 1 if t == len(sentence)-1 else backward_matrix[t][y_t]
        
        return (alpha*psi*beta) / Z
    
    
    
    
    # Exercise 1 d) ###################################################################
    def expected_feature_count(self, sentence, feature) -> float:
        '''
        Compute the expected feature count for the feature referenced by 'feature'
        Parameters: sentence: list of strings representing a sentence.
                    feature: a feature; element of the set 'self.features'
        Returns: float;
        Ek⃗θ (⃗x) = ∑Tt=1∑y,y′fk(y, y′, ⃗x)p(yt = y, yt−1 = y′|⃗x)
        '''
        expected_count = 0.0
        word = sentence[0][0]

        for tag in self.labels:
            active_features = self.get_active_features(word, tag, 'start')
            if feature in active_features:
                expected_count += self.marginal_probability(sentence, 0, tag, 'start')


        for t in range(1, len(sentence)):
            word = sentence[t][0]
            for tag in self.labels:
                for prev_tag in self.labels:
                    activeFeatures = self.get_active_features(word, tag, prev_tag)
                    if feature in activeFeatures:
                        expected_count += self.marginal_probability(sentence, t, tag, prev_tag)

        return expected_count
    
    
    # Exercise 1 e) ###################################################################
    def train(self, num_iterations, learning_rate=0.01):
        '''
        Method for training the CRF.
        Parameters: num_iterations: int; number of training iterations
                    learning_rate: float
        '''
        empirical_counts = [np.zeros(len(self.features)) for x in range(len(self.corpus))]
        
        print("Computing empirical counts...")
        for i in range(len(self.corpus)):
            sentence = self.corpus[i]
            for t in range(len(sentence)):
                word = sentence[t][0]
                tag = sentence[t][1]
                prev_tag = 'start' if t==0 else sentence[t-1][1]

                empirical_counts[i] += self.empirical_feature_count(word, tag, prev_tag)
        print("...End of computing empirical counts")

        print("Training started...")
        for i in range(num_iterations):
            print("Training on iteration", i+1)
            random_sentence_index = random.choice(range(len(self.corpus)))
            random_sentence = self.corpus[random_sentence_index]

            expected_counts = np.zeros(len(self.features))
            

            for feature in self.features.values():
                expected_counts[feature] = self.expected_feature_count(random_sentence, feature)

            empirical_count = empirical_counts[random_sentence_index]
            self.theta = self.theta + learning_rate * (expected_counts - empirical_count)

            self.current_forward_matrix = list()
            self.current_backward_matrix = list()
        
    

    
    
    
    
    # Exercise 2 ###################################################################
    def most_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of lables; each label is an element of the set 'self.labels'
        '''
        delta = [[None for pair in range(len(sentence))] for label in range(len(self.labels))]
        gamma = [[None for pair in range(len(sentence))] for label in range(len(self.labels))]
        sequence = [None for pair in range(len(sentence))]

        labels = list(self.labels)

        '''Initialization'''
        for t in range(len(labels)):
            label = labels[t]
            word = sentence[0][0]
            delta[t][0] = np.log(self.psi(word, label, 'start'))
        
        '''Recursion'''
        for t in range(1, len(sentence)):
            for j in range(len(labels)):
                max_val = 0.0
                max_index = -1
                for k in range(len(labels)):
                    word = sentence[t][0]
                    tag = labels[j]
                    prev_tag = labels[k]
                    prev_delta = delta[k][t-1]
                    value = np.log(self.psi(word, tag, prev_tag)) + prev_delta
                    
                    if value > max_val:
                        max_val = value
                        max_index = k

                delta[j][t] = max_val
                gamma[j][t] = max_index

        '''Total'''
        last = [delta[k][len(sentence)-1] for k in range(len(labels))]
        next = last.index(max(last))
        sequence[len(sentence)-1] = labels[next]

        for i in range(len(sentence)-1, 0, -1):
            next = gamma[next][i]
            sequence[i-1] = labels[next]

        return sequence

