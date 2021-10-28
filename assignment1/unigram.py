from generate_word import generate_word

"""Unigram distribution"""
def unigram_distribution(word_list:list):
    unigram_dict = generate_unigram(word_list)
    total_words_frequency = get_total_words(unigram_dict)
    unigrams_probability = calc_unigram_distribution(unigram_dict,total_words_frequency)
    return unigrams_probability,unigram_dict


"""Generate the dict with word frequency in the list"""
def generate_unigram(word_list:list) -> dict:
    unigram = {}
    for word in word_list:
        if word in unigram:
            unigram[word] = unigram[word] + 1
        else:
            unigram[word] = 1

    return unigram

"""Total word in corpus"""
def get_total_words(word_dict:dict) -> int:
    total_words = 0
    for word in word_dict:
        total_words = total_words + word_dict[word]
    return total_words

"""For each word calculate the unigram distribution according his frequency and total_words_frequency"""
def calc_unigram_distribution(word_dict:dict,total_words_frequency:int) -> dict:
    unigrams_probability = {}
    for word in word_dict.keys():
        word_frequency = word_dict[word]
        unigrams_probability[word] = word_frequency/total_words_frequency
    return unigrams_probability


"""Generating unigram sentence. Generate sample word without any filter until end of sentence tag."""
def unigram_gen_sentence(unigram_dict:list):
    generated_unigram=""
    start = "begin_of_sentence"
    end = "end_of_sentence"
    w = ("",0)

    while end not in w[0]:
        w = generate_word(unigram_dict)
        if w[0] != start: #ignoring the start sentences symbol
            generated_unigram += " " + w[0]

    generated_unigram += "\n"
    generated_unigram = generated_unigram.replace(" end_of_sentence",". end_of_sentence")

    print ("Unigrams sentence generated:")
    print (generated_unigram)


    

