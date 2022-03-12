################################################################################
## SNLP exercise sheet 5 Riccardo Preite 4196104
################################################################################
import os, random, sys
import numpy as np

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from tqdm import tqdm

class LdaModel(object):
    # add some instance variables for storing the corpus
    corpus: dict = None # [document1:[word1,...,wordn],..,documentM:[word1,...,wordk]]
    topics: list = None # List of topics name
    tokens: list = None # Unique tokens
    document_topic_total: list = None # [document1:[n_topic1,..,n_topic_N],...,documentM:[n_topic1,..,n_topic_N]]

    topic_word: list = None  # Number of topic for that word {word:topic[1,..,topic_number]}
    document_topic: list = None # number of topic in each document
    topic_count: list = None # total count of topic
    N: int = 0 # total number of words with repetition


    # Exercise 1 ###################################################################
    def initialize(self, path_to_corpus: str, percentage_corpus: int):
        '''
        Import and preprocess the corpus.
        Parameters: path_to_corpus: string; path to the directory containing the corpus
        '''
        self.corpus = list()
        self.topic_word = dict()
        self.document_topic_total = list()
        self.tokens = set()
        self.document_topic = list()
        self.filenames = list()

        self.topics = os.listdir(path_to_corpus)
        self.topic_count = np.array([0] * len(self.topics),dtype=np.int64)
        
        print("Initializing corpus with percentage", percentage_corpus)
        self.preprocess(path_to_corpus, percentage_corpus)  
        self.topic_counter()
                
    def preprocess(self, path_to_corpus: str, percentage_corpus):
        for topic in self.topics:
            self.remove_meta_information_from(path_to_corpus, topic, percentage_corpus)

    def topic_counter(self):
        for document in self.document_topic_total:
            self.topic_count += document
    
    # Exercise 2 ###################################################################
    def gibbs_sampling(self, num_iterations, alpha=0.25, beta=0.1):
        '''
        Implement the LDA Gibbs sampling algorithm.
        Parameters: num_iterations: int; number of sampling steps to do for each word
                    alpha: float; alpha parameter of the Dirichlet prior distribution
                    beta: float; beta parameter of the Dirichlet prior distribution
        '''
        pbar = tqdm(total=self.N*num_iterations,desc="Gibbs sampling")
        for _ in range(num_iterations):
            for i, (document_words, document_topics) in enumerate(zip(self.corpus, self.document_topic)):
                for j, (word_document, topic_document) in enumerate(zip(document_words, document_topics)):
                    self.document_topic_total[i][topic_document] -= 1
                    self.topic_word[word_document][topic_document] -= 1
                    self.topic_count[topic_document] -= 1

                    distribution = []
                    for topic_index, topic in enumerate(self.topics):
                        alpha_value = self.document_topic_total[i][topic_index] + alpha
                        beta_value = self.topic_word[word_document][topic_index] + beta
                        beta_len = beta * len(self.tokens)
                        normalization = self.topic_count[topic_index] + beta_len

                        p_zk = alpha_value * (beta_value/ normalization)
                        distribution.append(p_zk)
                        
                    new_topic = random.choices(range(len(self.topics)), distribution)[0]
                    self.document_topic[i][j] = new_topic
                    
                    self.document_topic_total[i][new_topic] += 1
                    self.topic_word[word_document][new_topic] += 1
                    self.topic_count[new_topic] += 1
                    pbar.update(1)
      

    # Utils
    def remove_meta_information_from(self, path_to_corpus: str, topic: str, percentage_corpus:int):
        topic_path = os.path.join(path_to_corpus,topic)
        file_list = os.listdir(topic_path)
        topic_to_take = (len(file_list)*percentage_corpus)//100

        for file_index, filename in tqdm(enumerate(file_list),desc=topic, total=topic_to_take):
            if file_index == topic_to_take:
                break

            path = os.path.join(topic_path, filename)
            no_stop_tokenized = self.get_and_tokenize(path)
            self.random_topic(no_stop_tokenized)

            self.corpus.append(no_stop_tokenized)
            self.tokens.update(no_stop_tokenized)
                          

    def get_and_tokenize(self, path):
        self.filenames.append(path)
        document = open(path, 'r')
        data = self.get_data(document)
        return self.tokenize(data)

    def get_data(self, document):
        meta_data = document.readlines()
        end_meta_index = meta_data.index("\n")
        data = [line.strip() for line in meta_data[end_meta_index+1:] if line.strip() != ""]
        string_data = " ".join(data)
        return string_data

    def tokenize(self, data):
        tokenized = word_tokenize(data)
        no_stop_tokenized = [word for word in tokenized if not word in stopwords.words('english') and word not in punctuation]
        self.N += len(no_stop_tokenized)
        return no_stop_tokenized

    def random_topic(self, tokenized):
        document_topic_total_counter = [0]*len(self.topics)
        document_topic_counter = [0]*len(tokenized)
        for token_index, token in enumerate(tokenized):
            topic_index = random.randint(0, len(self.topics)-1)                        
            if token not in self.topic_word:
                self.topic_word[token] = [0]*len(self.topics)
            
            document_topic_counter[token_index] = topic_index
            document_topic_total_counter[topic_index] += 1
            self.topic_word[token][topic_index] += 1
            
        self.document_topic.append(document_topic_counter)
        self.document_topic_total.append(document_topic_total_counter) 
        
    def predict_topic(self):
        text = ""
        for document_name, document_topic_count in zip(self.filenames, self.document_topic_total):
            converted = np.array(document_topic_count)
            text += "File: " + document_name + ", topic: " + str(converted.argmin())+"\n"
        out = open("prediction.txt","w+")
        out.write(text)

if __name__ == "__main__":
    percentage_corpus = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    if percentage_corpus <=0 or percentage_corpus >100:
        print("Invalid percentage, using now 100%")
        percentage_corpus = 100

    lda = LdaModel()
    lda.initialize("corpus/",percentage_corpus)

    print("Document number",len(lda.corpus))
    print("Total unique tokens",len(lda.tokens))
    print("Topic count",lda.topic_count)
    
    lda.gibbs_sampling(5)

    print("Topic count after sample", lda.topic_count)
    lda.predict_topic()
