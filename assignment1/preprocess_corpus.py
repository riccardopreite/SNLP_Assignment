# Riccardo Preite 4196104

from io import TextIOWrapper

"""Remove punctuaction from line, .\n is used to remove the point at the end of the sentence and not the point of the word (like Mr.)"""
def remove_punctuation(line:str) -> str:
    line=line.replace(".\n", "")
    line=line.replace("\n", "")
    line=line.replace("-", "")
    line=line.replace("+", "")
    line=line.replace("(", "")
    line=line.replace(")", "")
    line=line.replace("/", "")
    line=line.replace(",", "")
    line=line.replace(";", "")
    line=line.replace(":", "")
    line=line.replace("<", "")
    line=line.replace(">", "")
    line=line.replace("=", "")
    line=line.replace("?", "")
    line=line.replace("!", "")
    return line


def preprocessing_line(line_sentence: str):
    """Adding begin and end of sentence tag"""
    line_sentence = "BEGIN_OF_SENTENCE " + line_sentence + " END_OF_SENTENCE"
    """Remove punctuaction"""
    line_sentence=remove_punctuation(line_sentence)

    """Split and convert to lower case the words of the line"""
    line_splitted=line_sentence.split(" ")
    line_splitted = [word.lower() for word in line_splitted]
    
    return line_splitted


"""Iterate over corpus lines to preprocess them"""
def preprocess_corpus(corpus:TextIOWrapper) -> dict:
    words_list = []
    corpus_lines_list = corpus.readlines()
    for line in corpus_lines_list:
        words_list += preprocessing_line(line)

    return words_list
