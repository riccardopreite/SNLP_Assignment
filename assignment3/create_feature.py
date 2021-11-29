from typing import Tuple
import itertools


def get_word_and_tag_unique(corpus: list) -> Tuple[list,list,list]:
    word = []
    '''Initial tag'''
    tag = ['start']

    for sentence in corpus:
        for pair in sentence:
            word.append(pair[0])
            tag.append(pair[1])
    
   
    word_unique = list( set(word) )
    tag_unique = list( set(tag)  )
    return word_unique, tag_unique, tag

def build_feature_from_unique_list(word_unique: list,tag_unique: list) -> dict:
    feature = []
    # word_unique = ["the","dog"]
    # tag_unique = ["start","DN","NN"]

    word_tag_pair = list(itertools.product(word_unique, tag_unique))
    tag_tag_pair = list(itertools.product(tag_unique, tag_unique))

    feature.extend(word_tag_pair)
    feature.extend(tag_tag_pair)
    feature = set(feature)

    feature_dict = { pair:index for index, pair in enumerate(feature)   }
    return feature_dict
def create_feature(corpus: list)->Tuple[list, list]:
    
    word_unique, tag_unique, tag = get_word_and_tag_unique(corpus)

    return build_feature_from_unique_list(word_unique,tag_unique),tag_unique