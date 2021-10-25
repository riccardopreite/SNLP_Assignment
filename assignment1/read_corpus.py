
from os import name


CORPUS_NAME = "corpus.txt"

# Python function which
# takes a single string as input (representing a sentence) 
# and returns a sequence of words


def read_file(file_name: str) -> list:
    fd = open(file_name, 'r')
    lines_list = fd.readlines()

    return lines_list
    

def read_line(word_dict: dict,line_sentence: str) -> dict:
    line_splitted = []

    line_sentence.replace(",","")
    line_sentence.replace(":","")
    line_sentence.replace(";","")

    line_splitted = line_sentence.split(" ")

    line_splitted[0] = "BEGIN_OF_SENTENCE" + line_splitted[0]
    line_splitted[-1] = "END_OF_SENTENCE" + line_splitted[-1]

    for word in line_splitted:
        if word in word_dict:
            word_dict[word] = word_dict[word] + 1
        else:
            word_dict[word] = 0 

    return word_dict

def read_corpus() -> dict:
    corpus_lines_list = read_file(CORPUS_NAME)
    word = {}

    for line in corpus_lines_list:
        word = read_line(word,line)

    return word


def main():
    read_corpus()
    

if __name__ == "__main__":
    main()
