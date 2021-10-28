from generate_word import *

"""Trigram distribution"""
def trigram_distribution(word_list:dict,bigram_dict:dict):
    
    trigrams_dict = generate_trigram(word_list)
    trigrams_probability = calc_trigram_distribution(trigrams_dict,bigram_dict)
    return trigrams_probability,trigrams_dict


"""Generate the dict with word frequency in the list"""
def generate_trigram(word_list:list) -> dict:
    trigrams = {}
    for index,word in enumerate(word_list):
        if index > 1:
            prev1_word = word_list[index-1]
            prev2_word = word_list[index-2]
            trigram = (prev2_word,prev1_word, word)
            if trigram in trigrams:
                trigrams[trigram] = trigrams[trigram] + 1
            else:
                trigrams[trigram] = 1

    return trigrams

"""For each pair of three word calculate the trigram distribution according his frequency and the two previous words frequency from bigram"""

def calc_trigram_distribution(trigrams_dict:dict,bigram_dict:dict):
    trigrams_probability = {}
    for word in trigrams_dict.keys():
        word_frequency = trigrams_dict[word]
        bi_tuple = (word[0],word[1])
        trigrams_probability[word] = word_frequency/bigram_dict[bi_tuple]
    return trigrams_probability

"""Generating trigram sentence. Generate sample word with a filter. Starting from start_of_sentence tag with bigram_dict,
 and two previous word with trigram dict."""

def trigram_gen_sentence(trigram_dict:list,bigram_dict:list):
    generated_trigram=""
    prev = "begin_of_sentence"
    isFirst = True
    actual_generated = ""

    while "end_of_sentence" not in actual_generated:
        if isFirst:
            """Filt only on begin of sentence"""
            filtered_list = filt_list(prev,bigram_dict,True)
            actual_generated = generate_word(filtered_list)
            actual_generated = actual_generated[0][1]
            isFirst = False
        else:
            """Filt on w[i-2] and w[i-1]"""
            filtered_list = filt_list((prev,actual_generated),trigram_dict,False)
            prev = actual_generated
            actual_generated = generate_word(filtered_list)
            actual_generated = actual_generated[0][2]

        generated_trigram += " " + actual_generated

    generated_trigram += "\n"
    generated_trigram = generated_trigram.replace(" end_of_sentence",". end_of_sentence")
    print ("Trigrams sentence generated:")
    print (generated_trigram)


