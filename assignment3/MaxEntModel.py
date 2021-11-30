import math
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

    active_features = None
    empirical_feature_counts = None


    # Exercise 1 a) ###################################################################

    def initialize(self, corpus):
        '''
        Initialize the maximun entropy model, i.e., build the set of all features, the set of all labels
        and create an initial array 'theta' for the parameters of the model.
        Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
        '''
        self.active_features = dict()
        self.empirical_feature_counts = dict()
        self.corpus = corpus
        self.feature_indices, self.labels = create_feature(corpus)
        self.theta = np.ones(max(self.feature_indices.values())+1)       
        # your code here

    # Exercise 1 b) ###################################################################

    def get_active_features(self, word, label, prev_label) -> np.ndarray:
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

    def cond_normalization_factor(self, word, prev_label) -> float:
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

    def conditional_probability(self, label, word, prev_label) -> float:
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

        # your code here

    '''
    Implement the method empirical_feature_count, which returns the vector E[ ⃗ f(xi; yi)] of the empirical
    feature count, and the method expected_feature_count, which returns the vector E⃗[ ⃗ f(xi)]
    of the expected feature count, given the parameters ⃗ of the current model.
    '''

    # Exercise 3 a) ###################################################################
    def empirical_feature_count(self, word, label, prev_label) -> np.ndarray:
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

    def expected_feature_count(self, word, prev_label) -> np.ndarray:
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

    def parameter_update(self, word, label, prev_label, learning_rate):
        '''
        Do one learning step.
        Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                    label: string; the actual label of the selected word
                    prev_label: string; the label of the word at position i-1
                    learning_rate: float
        '''

        # your code here

        pass

    # Exercise 4 b) ###################################################################

    def train(self, number_iterations, learning_rate=0.1):
        '''
        Implement the training procedure.
        Parameters: number_iterations: int; number of parameter updates to do
                    learning_rate: float
        '''

        # your code here

        pass

    # Exercise 4 c) ###################################################################

    def predict(self, word, prev_label):
        '''
        Predict the most probable label of the word referenced by 'word'
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: string; most probable label
        '''

        # your code here

        pass

    # Exercise 5 a) ###################################################################

    def empirical_feature_count_batch(self, sentences):
        '''
        Predict the empirical feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the empirical feature count
        '''

        # your code here

    # Exercise 5 a) ###################################################################
    def expected_feature_count_batch(self, sentences):
        '''
        Predict the expected feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the expected feature count
        '''

        # your code here

    # Exercise 5 b) ###################################################################
    def train_batch(self, number_iterations, batch_size, learning_rate=0.1):
        '''
        Implement the training procedure which uses 'batch_size' sentences from to training corpus
        to compute the gradient.
        Parameters: number_iterations: int; number of parameter updates to do
                    batch_size: int; number of sentences to use in each iteration
                    learning_rate: float
        '''

        # your code here

        pass
