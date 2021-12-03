# Riccardo Preite 4196104
from typing import Tuple
import numpy as np
import random

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

class MaxEntModel(object):
    # training corpus
    corpus: list = list()

    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta: np.ndarray = np.array([])

    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    feature_indices: dict = dict()

    # set containing a list of possible lables
    # has to be set by the method 'initialize'
    labels: set = set()
    
    # Vector used to coun number of word in train and train batch respectivly
    word_counter_vector: list = list()
    word_batch_counter_vector: list = list()

    # Dict to store index of an active feature or empirical feature (word, tag, previous_tag)
    active_features: dict = dict()
    empirical_features: dict = dict()


    # Exercise 1 a) ###################################################################
    def initialize(self, corpus: list):
        '''
        Initialize the maximun entropy model, i.e., build the set of all features, the set of all labels
        and create an initial array 'theta' for the parameters of the model.
        Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
        '''
        
        self.corpus = corpus

        self.labels = set()
        self.active_features = dict()
        self.empirical_features = dict()
        self.feature_indices = dict()

        
        self.labels, self.feature_indices = create_feature_indices_and_tags(self.corpus)    
        self.theta = np.ones(len(self.feature_indices))
        print('Model initialized with a theta of len:',len(self.theta))

    # Exercise 1 b) ###################################################################
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
                self.feature_indices.keys()
            )
        )
        active_features: list = list(set([self.feature_indices[key] for key in keys_list]))
        
        self.active_features[active_feature_key] = np.array(active_features)
        
        return self.active_features[active_feature_key]

    # Exercise 2 a) ###################################################################
    def cond_normalization_factor(self, word: str, prev_label: str) -> float:
        '''
        Compute the normalization factor 1/Z(x_i).
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''

        Z = 0.0
        for label in self.labels:
            active_feature: np.ndarray = self.get_active_features(word, label, prev_label)
            vector_sum: float = 0

            for active_feature_index in active_feature:
                vector_sum += self.theta[active_feature_index]

            Z = np.exp(vector_sum)

        normalization: float = 1/Z
        return normalization

    # Exercise 2 b) ###################################################################
    def conditional_probability(self, word: str, label: str, prev_label: str) -> float:
        '''
        Compute the conditional probability of a label given a word x_i.
        Parameters: label: string; we are interested in the conditional probability of this label
                    word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''
        normalization: float = self.cond_normalization_factor(word, prev_label)
        active_feature: np.ndarray = self.get_active_features(word, label, prev_label)
        vector_sum = 0
        for active_feature_index in active_feature:
            vector_sum += self.theta[active_feature_index]
            
        conditional_probablity: float = normalization * np.exp(vector_sum)
        return conditional_probablity

    # Exercise 3 a) ###################################################################
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
            
        empirical_theta: np.ndarray = np.zeros(len(self.feature_indices))
        active_feature:np.ndarray = self.get_active_features(word, label, prev_label)

        for active_feature_index in active_feature:
            empirical_theta[active_feature_index] = 1
            
        self.empirical_features[empirical_feature_key] = empirical_theta
        return self.empirical_features[empirical_feature_key]


    # Exercise 3 b) ###################################################################expected
    def expected_feature_count(self, word: str, prev_label: str) -> np.ndarray:
        '''
        Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
        (see variable theta)
        Parameters: word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the expected feature count
        '''
                
        expected_theta: np.ndarray = np.zeros(len(self.feature_indices))

        for label in self.labels:
            conditional_probability_with_actual_label: float = self.conditional_probability(word, label, prev_label)
            active_feature_with_actual_label: np.ndarray = self.get_active_features(word, label, prev_label)

            for active_feature_index in active_feature_with_actual_label:
                expected_theta[active_feature_index] += conditional_probability_with_actual_label

        return expected_theta

    # Exercise 4 a) ###################################################################
    def parameter_update(self, word: str, label: str, prev_label: str, learning_rate: float):
        '''
        Do one learning step.
        Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                    label: string; the actual label of the selected word
                    prev_label: string; the label of the word at position i-1
                    learning_rate: float
        '''
        gradient: float = self.empirical_feature_count(word, label, prev_label) - self.expected_feature_count(word, prev_label)
        self.theta = self.theta + learning_rate * gradient



    # Exercise 4 b) ###################################################################
    def train(self, number_iterations: int, learning_rate: float=0.1):
        '''
        Implement the training procedure.
        Parameters: number_iterations: int; number of parameter updates to do
                    learning_rate: float
        '''
        for iteration in range(number_iterations):
            random_sentence_index: int = random.randint(0, len(self.corpus) - 1)
            random_word_index: int = random.randint(0, len(self.corpus[random_sentence_index]) - 1)
            word_tag: tuple = self.corpus[random_sentence_index][random_word_index]
            
            word = word_tag[0]
            label = word_tag[1]
            prev_label = "start" if random_word_index == 0 else self.corpus[random_sentence_index][random_word_index-1]

            if len(self.word_counter_vector) == 0:
                self.word_counter_vector.append(1)
            else:
                self.word_counter_vector.append(self.word_counter_vector[-1]+1)
                
            self.parameter_update(word, label, prev_label, learning_rate)

    # Exercise 4 c) ###################################################################
    def predict(self, word: str, prev_label: str) -> bool:
        '''
        Predict the most probable label of the word referenced by 'word'
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: string; most probable label
        '''
        probability: np.ndarray = np.zeros(len(self.labels))
        tags: list = list(self.labels)
        for label_index in range(len(tags)):
            label: str = tags[label_index]
            probability[label_index] = self.conditional_probability(word, label, prev_label)
        
        return tags[np.argmax(probability)]
        
        
        # Exercise 5 a) ###################################################################
    def empirical_feature_count_batch(self, sentences: list) -> np.ndarray:
        '''
        Predict the empirical feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the empirical feature count
        '''
        
        empirical_features_array: np.ndarray = np.zeros(len(self.feature_indices))

        for sentence in sentences:
            for word_index, word_tag in enumerate(sentence):
                word = word_tag[0]
                tag = word_tag[1]
                prev_label = "start" if word_index == 0 else sentence[word_index-1][1]

                empirical_features_array += self.empirical_feature_count(word, tag, prev_label)
                
        return empirical_features_array         
        
    # Exercise 5 a) ###################################################################
    def expected_feature_count_batch(self, sentences: list) -> np.ndarray:
        '''
        Predict the expected feature count for a set of sentences
        Parameters: sentences: list; a list of sentences; should be a sublist of the list returnd by 'import_corpus'
        Returns: (numpy) array containing the expected feature count
        '''
        
        expected_features_array: np.ndarray = np.zeros(len(self.feature_indices))
        for sentence in sentences:
            for word_index, word_tag in enumerate(sentence):
                word = word_tag[0]
                prev_label = "start" if word_index == 0 else sentence[word_index-1][1]
                
                expected_features_array += self.expected_feature_count(word, prev_label)
                
        return expected_features_array
        
        
    
    
    
    
    # Exercise 5 b) ###################################################################
    def train_batch(self, number_iterations: int, batch_size: int, learning_rate: float=0.1):
        '''
        Implement the training procedure which uses 'batch_size' sentences from to training corpus
        to compute the gradient.
        Parameters: number_iterations: int; number of parameter updates to do
                    batch_size: int; number of sentences to use in each iteration
                    learning_rate: float
        '''
        for iteration in range(number_iterations):
            sentences = random.sample(self.corpus,k=batch_size)
            for sentence in sentences:
                if len(self.word_batch_counter_vector) == 0:
                    self.word_batch_counter_vector.append(len(sentence))
                else:
                    self.word_batch_counter_vector.append(self.word_batch_counter_vector[-1]+len(sentence))
                    
            gradient: float = (self.empirical_feature_count_batch(sentences) - self.expected_feature_count_batch(sentences))
            self.theta = self.theta + learning_rate * gradient