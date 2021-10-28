# Riccardo Preite 4196104
from preprocess_corpus import *
from unigram import *
from bigram import *
from trigram import *
from generate_word import sortDict
CORPUS_NAME = "corpus.txt"
ANSWER_NAME = "answer.txt"


"""Open corpus file and preprocess it"""

def read_corpus() -> dict:
    corpus = open(CORPUS_NAME, "r")
    words_list = preprocess_corpus(corpus)
    return words_list

def show_answer():
    answer = open(ANSWER_NAME, "r")
    print(answer.read())

def main():
    words_list = read_corpus()
    """Calculating unigram distribution"""
    unigram_probability,unigram_dict = unigram_distribution(words_list)
    # print(unigram_probability["the"])

    """Calculating bigram distribution"""
    bigram_probability,bigram_dict = bigram_distribution(words_list,unigram_dict)
    # print(bigram_probability[("the","amateur")])

    """Calculating trigram distribution"""
    trigram_probability,trigram_dict = trigram_distribution(words_list,bigram_dict)
    # print(trigram_probability[("a", "skeleton", "corporation")])

    """Sorting dict in decrease order to choose word with higher prob"""
    unigram_prob_sorted = sortDict(unigram_probability)
    bigram_prob_sorted = sortDict(bigram_probability)
    trigram_prob_sorted = sortDict(trigram_probability)

    print("##########################")
    print("START OF SENTENCE GEN\n\n")

    """Generating uni,bi,tri gram sentence"""
    unigram_gen_sentence(unigram_prob_sorted)
    bigram_gen_sentence(bigram_prob_sorted)
    trigram_gen_sentence(trigram_prob_sorted,bigram_prob_sorted)
    print("END OF SENTENCE GEN")
    print("##########################\n\n")
    show_answer()

    print("END OF SHOW ANSWER")
    print("##########################\n\n")
    


    

if __name__ == "__main__":
    main()
