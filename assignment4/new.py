import math
import sys
import random
import numpy as np
from typing import Tuple

from numpy.lib.function_base import gradient

def create_feature_indices_and_tags(corpus: list) -> Tuple[set, dict]:
            words_unique = set()
            tags_unique = set("start")
            feature_set = set()
            for sentence in corpus:
                words_unique.update(list(map(lambda tuple: tuple[0], sentence)))
                tags_unique.update(list(map(lambda tuple: tuple[1], sentence)))

            feature_set.update([(words, label) for words in words_unique for label in tags_unique])
            feature_set.update([(label1, label2) for label1 in tags_unique for label2 in tags_unique])
            
            feature_indices: dict = {feature: index for index, feature in enumerate(feature_set)}  

            feature_index = 0
            # feature_indices: dict = dict()
            # for feature in feature_set:
            #     feature_indices[feature] = feature_index
            #     feature_index += 1

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
    # set containing all lables observed in the corpus 'self.corpus'
    labels: set = set()

    active_features: dict = dict()
    empirical_features:dict = dict()

    current_forward_matrix: list = None
    current_backward_matrix: list = None
    
   
    
    def initialize(self, corpus):
        '''
        build set two sets 'self.features' and 'self.labels'
        '''
        self.corpus = corpus

        self.labels = set()
        self.active_features = dict()
        self.empirical_features = dict()
        self.features = dict()

        words = set()
        self.labels, self.features = create_feature_indices_and_tags(self.corpus)    
        # for sentence in self.corpus:
        #     word = list(map(lambda tuple: tuple[0], sentence))
        #     labels = list(map(lambda tuple: tuple[1], sentence))
        #     words.update(word)
        #     self.labels.update(labels)
        
        # features = set()
        # w_t_feature = [(word, label) for word in words for label in self.labels]
        # t_t_feature = [(prev_label, label) for prev_label in self.labels for label in self.labels]
        # s_t_feature = [('start', label) for label in self.labels]

        # features.update(w_t_feature)
        # features.update(t_t_feature)
        # features.update(s_t_feature)

        # index = 0
        # for feature in features:
        #     self.features[feature] = index
        #     index += 1
        
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
        
        active_features = set()
        for key in self.features.keys():
            word_f = key[0]
            tag_f = key[1]
            feature_index = self.features[key]
            if (word == word_f and label == tag_f) or (prev_label == word_f and label == tag_f):
                active_features.add(feature_index)

        self.active_features[active_feature_key] = active_features      
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


    def mpsi(self, word: str, label: str, prev_label: str):
        active_features = self.get_active_features(word, label, prev_label)
        summed = sum(map(lambda i: self.theta[i], active_features))
        return np.exp(summed)

    def psi(self, label, prevLabel, word):
        activeFeatures = self.get_active_features(word, label, prevLabel)
        expArg = sum(map(lambda i: self.theta[i], activeFeatures))
        return np.exp(expArg)
        
    # Exercise 1 a) ###################################################################
    def mforward_variables(self, sentence: list) -> list:
        '''
        Compute the forward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of forward variables
        '''
        if self.current_forward_matrix is not None:
            return self.current_forward_matrix

        forward_matrix: list = [dict() for pair in range(len(sentence))]
        '''Initialization'''
        word: str = sentence[0][0]
        prev_label: str = "start"

        for tag in list(self.labels):
            forward_matrix[0][tag] = self.psi(word, tag, prev_label) 

        '''Recursion'''
        for t in range(1,len(sentence)):
            word_t = sentence[t][0]

            for tag_j in list(self.labels):
                new_alpha: float = 0.0
                for tag_i in list(self.labels):
                    psi = self.psi(word_t, tag_j, tag_i)
                    prevAlpha = forward_matrix[t-1][tag_i]
                    new_alpha += psi * prevAlpha
                forward_matrix[t][tag_j] = new_alpha

        self.current_forward_matrix = forward_matrix
        return forward_matrix        
    def forward_variables(self,sentence: list) -> list:
        if self.current_forward_matrix is not None:
            return self.current_forward_matrix
        forwardMatrix = [dict() for x in range(len(sentence))]
        labelsList = list(self.labels)

        # INIT
        for label in labelsList:
            word = sentence[0][0]
            prevLabel = 'start'
            psi = self.psi(label, prevLabel, word)
            forwardMatrix[0][label] = psi

        # INDUCTION
        for i in range(1, len(sentence)):
            for label in labelsList:
                word = sentence[i][0]
                sum = 0
                for prevLabel in labelsList:
                    psi = self.psi(label, prevLabel, word)
                    prevAlpha = forwardMatrix[i-1][prevLabel]
                    sum += psi * prevAlpha
                forwardMatrix[i][label] = sum

        self.current_forward_matrix = forwardMatrix
        return forwardMatrix
        
        
    def mbackward_variables(self, sentence) -> list:
        '''
        Compute the backward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of backward variables
        '''
        if self.current_backward_matrix is not None:
            return self.current_backward_matrix

        backward_matrix: list = [dict() for pair in range(len(sentence))]

        '''Initialization'''
        for tag in list(self.labels):
            backward_matrix[len(sentence)-1][tag] = 1
        
        '''Recursion'''
        for t in range(len(sentence)-2,-1,-1):
            word_t = sentence[t+1][0]

            for tag_j in list(self.labels):
                new_alpha: float = 0.0
                for tag_i in list(self.labels):
                    psi = self.psi(word_t, tag_i, tag_j)
                    alpha_prev = backward_matrix[t+1][tag_i]
                    new_alpha += psi * alpha_prev
                
                backward_matrix[t][tag_j] = new_alpha

        self.current_backward_matrix = backward_matrix
        return backward_matrix

    def backward_variables(self, sentence):
        '''
        Compute the backward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of backward variables
        '''
        if self.current_backward_matrix is not None:
            return self.current_backward_matrix

        backwardMatrix = [dict() for x in range(len(sentence))]
        labelsList = list(self.labels)

        # INIT
        lastIndex = len(sentence)-1
        for label in labelsList:
            backwardMatrix[lastIndex][label] = 1

        # INDUCTION
        for i in range(len(sentence)-2, -1, -1):
            for prevLabel in labelsList:
                word = sentence[i+1][0]
                sum = 0
                for label in labelsList:
                    psi = self.psi(label, prevLabel, word)
                    prevBeta = backwardMatrix[i+1][label]
                    sum += psi * prevBeta
                backwardMatrix[i][prevLabel] = sum

        self.current_backward_matrix = backwardMatrix
        return backwardMatrix
        
        
        
    
    # Exercise 1 b) ###################################################################
    def mcompute_z(self, sentence) -> float:
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        backward_matrix = self.backward_variables(sentence)
        word = sentence[0][0]
        prev_label = 'start'

        z = 0
        for i in range(len(list(self.labels))):
            label = list(self.labels)[i]
            psi = self.psi(label, prev_label, word)
            beta = backward_matrix[i][0]
            z += psi * beta
        '''
        
        forward_matrix: np.ndarray = self.forward_variables(sentence)
        Z: float = 0.0
        last = len(sentence)-1
        for label in list(self.labels):
            Z += forward_matrix[last][label]

        return Z

    def compute_z(self, sentence):
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        '''
        '''
        backwardMatrix = self.backward_variables(sentence)
        labelsList = list(self.labels)
        word = sentence[0][0]
        prevLabel = 'start'

        z = 0
        for i in range(len(labelsList)):
            label = labelsList[i]
            psi = self.psi(label, prevLabel, word)
            betaOne = backwardMatrix[i][0]
            z += psi * betaOne
        '''
        
        forwardMatrix = self.forward_variables(sentence)
        labelsList = list(self.labels)
        lastWordIndex = len(sentence)-1

        z = 0
        for label in labelsList:
            z += forwardMatrix[lastWordIndex][label]
        
        return z   
        
            
    # Exercise 1 c) ###################################################################
    def mmarginal_probability(self, sentence: list, t:int, y_t: str, y_t_minus_one: str) -> float:
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
        Returns: float: probability;
        '''
        norm_Z: float = 1 / self.compute_z(sentence)
        forward_matrix: np.ndarray = self.forward_variables(sentence)
        backward_matrix: np.ndarray = self.backward_variables(sentence)
        
        word = sentence[t][0]

        psi = self.psi(word, y_t, y_t_minus_one)
        alpha = 1 if t == 0 else forward_matrix[t-1][y_t_minus_one]
        beta = 1 if t == len(sentence)-1 else backward_matrix[t][y_t]
        
        return alpha*psi*beta*norm_Z
    
    def marginal_probability(self, sentence, t, y_t, y_t_minus_one):
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
                    t: int; position of the word the label y_t is assigned to
        Returns: float: probability;
        '''
        
        zetaNorm = 1 / self.compute_z(sentence)
        forwardMatrix = self.forward_variables(sentence)
        backwardMatrix = self.backward_variables(sentence)
        
        prob = 0.0
        
        word = sentence[t][0]
        psi = self.psi(y_t, y_t_minus_one, word)
        alpha = 1 if t == 0 else forwardMatrix[t-1][y_t_minus_one]
        beta = 1 if t == len(sentence)-1 else backwardMatrix[t][y_t]
        prob = zetaNorm * alpha * psi * beta

        return prob
    
    
    # Exercise 1 d) ###################################################################
    def mexpected_feature_count(self, sentence, feature) -> float:
        '''
        Compute the expected feature count for the feature referenced by 'feature'
        Parameters: sentence: list of strings representing a sentence.
                    feature: a feature; element of the set 'self.features'
        Returns: float;
        Ek⃗θ (⃗x) = ∑Tt=1∑y,y′fk(y, y′, ⃗x)p(yt = y, yt−1 = y′|⃗x)
        '''
        expected_count = 0.0
        word = sentence[0][0]

        for tag in list(self.labels):
            active_features = self.get_active_features(word, tag, 'start')
            if feature in active_features:
                expected_count += self.marginal_probability(sentence, 0, tag, 'start')


        for t in range(1, len(sentence)):
            word = sentence[t][0]
            for tag in list(self.labels):
                for prev_tag in list(self.labels):
                    active_features = self.get_active_features(word, tag, prev_tag)
                    if feature in active_features:
                        expected_count += self.marginal_probability(sentence, t, tag, prev_tag)

        return expected_count
    
    def expected_feature_count(self, sentence, feature):
        '''
        Compute the expected feature count for the feature referenced by 'feature'
        Parameters: sentence: list of strings representing a sentence.
                    feature: a feature; element of the set 'self.features'
        Returns: float;
        '''
        count = 0.0
        labelsList = list(self.labels)

        # start case
        word = sentence[0][0]
        for label in labelsList:
            activeFeatures = self.get_active_features(word, label, 'start')
            if feature in activeFeatures:
                count += self.marginal_probability(sentence, 0, label, 'start')

        for t in range(1, len(sentence)):
            word = sentence[t][0]
            for label in labelsList:
                for prevLabel in labelsList:
                    activeFeatures = self.get_active_features(word, label, prevLabel)
                    if feature in activeFeatures:
                        count += self.marginal_probability(sentence, t, label, prevLabel)

        return count
    
    # Exercise 1 e) ###################################################################
    def mtrain(self, num_iterations, learning_rate=0.01):
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

        print("Training started with Num_iteration:",num_iterations,"...")
        for i in range(num_iterations):
            print("\tTraining on iteration", i+1)
            random_sentence_index = random.choice(range(len(self.corpus)))
            random_sentence = self.corpus[random_sentence_index]

            expected_counts = np.zeros(len(self.features))
            

            for feature in self.features.values():
                expected_counts[feature] = self.expected_feature_count(random_sentence, feature)

            empirical_count = empirical_counts[random_sentence_index]
            gradient: float = expected_counts - empirical_count
            back = self.theta
            self.theta = self.theta + learning_rate * gradient
            print("new theta:", self.theta, "\n")
            print("comp:",(back==self.theta).all())
            self.current_forward_matrix = None
            self.current_backward_matrix = None
        
    def train(self, num_iterations, learning_rate=0.01):
        '''
        Method for training the CRF.
        Parameters: num_iterations: int; number of training iterations
                    learning_rate: float
        '''
        # Precomputing empirical counts
        computedEmCounts = [np.zeros(len(self.features)) for x in range(len(self.corpus))]

        print("Precomputing empirical counts...")
        for i in range(len(self.corpus)):
            sentence = self.corpus[i]
            for t in range(len(sentence)):
                word = sentence[t][0]
                label = sentence[t][1]
                prevLabel = sentence[t-1][1] if t > 0 else 'start'
                computedEmCounts[i] += self.empirical_feature_count(word, label, prevLabel)

        print("Training started...")
        for i in range(num_iterations):
            print("Iteration", i+1)
            randomIndex = random.choice(range(len(self.corpus)))
            randomSentence = self.corpus[randomIndex]
            exCounts = np.zeros(len(self.features))
            emCounts = computedEmCounts[randomIndex]

            for feature in self.features.values():
                exCounts[feature] = self.expected_feature_count(randomSentence, feature)

            self.theta = self.theta + learning_rate * (emCounts - exCounts)
            print("new theta:", self.theta, "\n")

            self.current_forward_matrix = None
            self.current_backward_matrix = None

    
    
    
    
    # Exercise 2 ###################################################################
    def mmost_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of lables; each label is an element of the set 'self.labels'
        '''
        delta = [[None for pair in range(len(sentence))] for label in range(len(self.labels))]
        gamma = [[None for pair in range(len(sentence))] for label in range(len(self.labels))]
        sequence = [None for pair in range(len(sentence))]

        labels = list(self.labels)
        word = sentence[0][0]
        
        '''Initialization'''
        for t in range(len(labels)):
            label = labels[t]    
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

    def most_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of lables; each label is an element of the set 'self.labels'
        '''
        deltaMatrix = [[None for x in range(len(sentence))] for y in range(len(self.labels))]
        gammaMatrix = [[None for x in range(len(sentence))] for y in range(len(self.labels))]
        sequenceTagging = [None for x in range(len(sentence))]

        labelsList = list(self.labels)

        ## INIT
        for i in range(len(labelsList)):
            label = labelsList[i]
            word = sentence[0][0]
            deltaMatrix[i][0] = np.log(self.psi(label, 'start', word))

        ## INDUCTION
        for i in range(1, len(sentence)):
            for j in range(len(labelsList)):
                maxVal = 0.0
                maxValIndex = -1
                for k in range(len(labelsList)):
                    word = sentence[i][0]
                    label = labelsList[j]
                    prevLabel = labelsList[k]
                    precDelta = deltaMatrix[k][i-1]
                    value = np.log(self.psi(label, prevLabel, word)) + precDelta
                    
                    if value > maxVal:
                        maxVal = value
                        maxValIndex = k
                
                deltaMatrix[j][i] = maxVal
                gammaMatrix[j][i] = maxValIndex

        ## TOTAL
        lastCol = [deltaMatrix[k][len(sentence)-1] for k in range(len(labelsList))]
        nextIndex = lastCol.index(max(lastCol))
        sequenceTagging[len(sentence)-1] = labelsList[nextIndex]

        for i in range(len(sentence)-1, 0, -1):
            nextIndex = gammaMatrix[nextIndex][i]
            sequenceTagging[i-1] = labelsList[nextIndex]

        return sequenceTagging
