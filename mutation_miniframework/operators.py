from mutation_miniframework.base_mutators import *

#Mutation Operators
'''
Saves new file output with added functional name additions
'''


def replaceFromDictionary(dataset : Dataset, word_list : dict, mutation="misspell", word_change_limit=1):
    replaceWords(dataset, word_list, word_change_limit)
    dataset.saveDataMutation(mutation)
'''
Takes an article and adds spaces between to replace
'''
def replaceArticles(dataset : Dataset, articles : dict, mutation="articleSub", word_change_limit=1):
    replaceWords(dataset, articles, word_change_limit)
    dataset.saveDataMutation(mutation)

def replaceLetters(dataset : Dataset, articles : dict, mutation="letterReplace", word_change_limit=1):
    replaceWords(dataset, articles, word_change_limit)
    dataset.saveDataMutation(mutation)

def replaceSynonyms(dataset : Dataset, words_to_replace : list, mutation="synonymSub", word_change_limit=1):
    word_list = {}
    for word in words_to_replace:
        word_list[word] = getSynonymAPI(word)
    replaceWords(dataset, word_list, word_change_limit)
    dataset.saveDataMutation(mutation)

def replaceInTextsRandomSynonymAPI(dataset : Dataset, mutation="randSynonym", word_change_limit=1):
    word_list = {}
    for key, texts in dataset.items():
        for word in texts.split(" "):
            word_list[word] = getSynonymAPI(word)
    replaceWords(dataset, word_list, word_change_limit)
    dataset.saveDataMutation(mutation)

def replaceInTextsAntonymAPI(dataset : Dataset, words_to_replace : list, mutation="antonym", word_change_limit=1):
    word_list = {}
    for word in words_to_replace:
        word_list[word] = getAntonymAPI(word)
    replaceWords(dataset, word_list, word_change_limit)
    dataset.saveDataMutation(mutation)

def replaceInTextsRandomAntonymAPI(dataset : Dataset, mutation="randAntonym", word_change_limit=1):
    word_list = {}
    for key, texts in dataset.items():
        for word in texts.split(" "):
            word_list[word] = getAntonymAPI(word)
    replaceWords(dataset, word_list, word_change_limit)
    dataset.saveDataMutation(mutation)

'''
Replaces an adjective with another
'''
def replaceInTextsRandomAdjective(dataset : Dataset, word_change_limit=1):
    pass

'''
Replaces a verb with another
'''
def replaceInTextsRandomVerb(dataset : Dataset, word_change_limit=1):
    pass

def removeStartingArticles(dataset : Dataset):
    for key, texts in dataset.items():
        textsBuffer = []
        for text in texts:
            text : str
            text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            wordsList = text.split(" ")
            if wordsList[0].lower() == "a" or wordsList[0].lower() == "the" or wordsList[0].lower() == "an":
                wordsList = wordsList[1:]
            text = ' '.join(wordsList)
            textsBuffer.append(text)
        dataset[key] = textsBuffer

def deleteRandomArticle(dataset : Dataset, articles : list, mutation="delArticles", word_change_limit=1):
    word_list = {}
    for article in articles:
        word_list[article] = " "
    replaceWords(dataset, word_list, word_change_limit)
    removeStartingArticles(dataset)
    dataset.saveDataMutation(mutation)

def replaceWordListWithRandomSelf(dataset : Dataset, random_words : list, mutation="randWord", word_change_limit=1):
    replaceWordsByDoubleList(dataset, random_words, word_change_limit)
    dataset.saveDataMutation(mutation)