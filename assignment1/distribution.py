from read_corpus import read_corpus

def calc_distribution_all_words(words: dict,total_words:int) -> int :
    distribution = 0

    return distribution

def calc_unigram_distribution(words: dict,total_words:int,word1: str,word2: str) -> int :
    distribution = 0
    if word1 in words and word2 in words:
        P_w1 = words[word1]/(total_words)
        P_w2 = words[word2]/(total_words)
        print("FW1 ", words[word1])
        print("FW2 ", words[word2])
        print("Probability 1 ", P_w1)
        print("Probability 2 ", P_w2)
        distribution = P_w1 * P_w2

    return distribution

def calc_bigram_distribution(words: dict,total_words:int,word1: str,word2: str,word3: str) -> int :
    distribution = 0

    return distribution



def main():
    word_dict = read_corpus()
    total_words = 0
    for word in word_dict:
        total_words = total_words + word_dict[word]

    # distribution1 = calc_distribution_all_words(word_dict,total_words)
    unigram_distribution = calc_unigram_distribution(word_dict,total_words,"with","The")
    print(unigram_distribution)
    # distribution3 = calc_bigram_distribution(word_dict,total_words)
    

if __name__ == "__main__":
    main()