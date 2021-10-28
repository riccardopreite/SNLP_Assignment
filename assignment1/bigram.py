from generate_word import *

"""Bigram distribution"""
def bigram_distribution(word_list:dict,unigram_dict:dict):
    
    bigrams_dict = generate_bigram(word_list)
    bigrams_probability = calc_bigram_distribution(bigrams_dict,unigram_dict)
    return bigrams_probability,bigrams_dict


"""Generate the dict with word frequency in the list"""
def generate_bigram(word_list:list) -> dict:
    bigrams = {}
    for index,word in enumerate(word_list):
        if index > 0:
            prev_word = word_list[index-1]
            bigram = (prev_word, word)
            if bigram in bigrams:
                bigrams[bigram] = bigrams[bigram] + 1
            else:
                bigrams[bigram] = 1

    return bigrams


"""For each pair of two word calculate the bigram distribution according his frequency and previous word frequency from unigram"""
def calc_bigram_distribution(bigrams_dict:dict,unigram_dict:dict)-> dict:
    bigrams_probability = {}
    for word in bigrams_dict.keys():
        word_frequency = bigrams_dict[word]
        bigrams_probability[word] = word_frequency/unigram_dict[word[0]]
    return bigrams_probability

"""Generating bigram sentence. Generate sample word with a filter. Starting from start_of_sentence tag and refilter with every new word generate from sample."""
def bigram_gen_sentence(bigram_dict:list):
    generated_bigram=""
    start = "begin_of_sentence"
    end = "end_of_sentence"

    """First word has to be choosen with begin of sentence tag"""
    filtered_list = filt_list(start,bigram_dict,True)

    w = (("",""),0)
    while end not in w[0]:
        
        w = generate_word(filtered_list)

        filtered_list = filt_list(w[0][1],bigram_dict,True)
        generated_bigram += " " + w[0][1]

    generated_bigram += "\n"
    generated_bigram = generated_bigram.replace(" end_of_sentence",". end_of_sentence")

    print ("Bigrams sentence generated:")
    print (generated_bigram)


