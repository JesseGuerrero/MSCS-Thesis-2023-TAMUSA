from mutation_operators.Dataset import *
from mutation_operators.base_mutators import *

#Mutation Operators
'''
Saves new file output with added functional name additions
'''
import random
def misspellWords(dataset : Dataset, word_list : dict, mutation="misspell", per_caption_limit=1):
    replaceWords(dataset, word_list, per_caption_limit)
    dataset.saveDataMutation(mutation)
'''
Takes an article and adds spaces between to replace
'''
def replaceArticles(dataset : Dataset, articles : dict, mutation="articleSub", per_caption_limit=1):
    replaceWords(dataset, articles, per_caption_limit)
    dataset.saveDataMutation(mutation)

def replaceSynonym(dataset : Dataset, words_to_replace : list, mutation="synonymSub", per_caption_limit=1):
    word_list = {}
    for word in words_to_replace:
        word_list[word] = getSynonymAPI(word)
    replaceWords(dataset, word_list, per_caption_limit)
    dataset.saveDataMutation(mutation)

def replaceInCaptionRandomSynonym(dataset : Dataset, mutation="randSynonym", per_caption_limit=1):
    word_list = {}
    for key, caption in dataset.items():
        for word in caption.split(" "):
            word_list[word] = getSynonymAPI(word)
    replaceWords(dataset, word_list, per_caption_limit)
    dataset.saveDataMutation(mutation)

def replaceInCaptionAntonym(dataset : Dataset, words_to_replace : list, mutation="antonym", per_caption_limit=1):
    word_list = {}
    for word in words_to_replace:
        word_list[word] = getAntonymAPI(word)
    replaceWords(dataset, word_list, per_caption_limit)
    dataset.saveDataMutation(mutation)

def replaceInCaptionRandomAntonym(dataset : Dataset, mutation="randAntonym", per_caption_limit=1):
    word_list = {}
    for key, caption in dataset.items():
        for word in caption.split(" "):
            word_list[word] = getAntonymAPI(word)
    replaceWords(dataset, word_list, per_caption_limit)
    dataset.saveDataMutation(mutation)

'''
Replaces an adjective with another
'''
def replaceInCaptionRandomAdjective(dataset : Dataset, per_caption_limit=1):
    pass

'''
Replaces a verb with another
'''
def replaceInCaptionRandomVerb(dataset : Dataset, per_caption_limit=1):
    pass

def deleteRandomArticle(dataset : Dataset, articles : list, mutation="delArticles", per_caption_limit=1):
    word_list = {}
    for article in articles:
        word_list[article] = " "
    replaceWords(dataset, word_list, per_caption_limit)
    dataset.saveDataMutation(mutation)