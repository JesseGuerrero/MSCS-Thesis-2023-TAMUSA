from mutation_operators.Dataset import *
#Base functions
'''
This is a base function, meant to assist the actual operators.
It returns (new addition to new file name, new data)
'''
def replaceSubString(dataset : Dataset, substring, replacement, count=-1):
    for key, captions in dataset.items():
        captionsBuffer = []
        for caption in captions:
            captions : str
            captionsBuffer.append(caption.replace(substring, replacement, count))
        dataset[key] = captionsBuffer

'''
This is a base function, meant to assist the actual operators.
It returns (new addition to new file name, new data)
'''
import re
def replaceWords(dataset : Dataset, word_list : dict, count=1):
    for key, captions in dataset.items():
        captionsBuffer = []
        for caption in captions:
            caption : str
            amountToReplace = count
            for word, replacement in word_list.items():
                (caption, numReplaced) = re.subn(word, replacement, caption, amountToReplace)
                amountToReplace -= numReplaced
                if amountToReplace <= 0:
                    break
            captionsBuffer.append(caption)
        dataset[key] = captionsBuffer

import requests, json
def getSynonymAPI(word) -> str:
    with open("./mutation_data/synonyms.json", "r+") as f:
        local_synonyms = json.load(f)
        if word in local_synonyms:
            if len(local_synonyms[word]) is 0:
                return word
            return local_synonyms[word][0]
        url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/synonyms"

        headers = {
            "X-RapidAPI-Key": "41b9c2ee17msh295225a18398362p1c732cjsn4afb01c0b61f",
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }

        response = requests.request("GET", url, headers=headers)
        print(dict(response.json()))
        synonyms = response.json()['synonyms']
        local_synonyms[word] = synonyms
        f.seek(0)
        f.write((json.dumps(local_synonyms)))
        f.truncate()
        if len(synonyms) is 0:
            return word
        return synonyms

def getAntonymAPI(word) -> str:
    with open("./mutation_data/antonyms.json", "r+") as f:
        local_antonyms = json.load(f)
        if word in local_antonyms:
            if len(local_antonyms[word]) is 0:
                return word
            return local_antonyms[word][0]
        url = f"https://wordsapiv1.p.rapidapi.com/words/{word}/antonyms"

        headers = {
            "X-RapidAPI-Key": "41b9c2ee17msh295225a18398362p1c732cjsn4afb01c0b61f",
            "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }

        response = requests.request("GET", url, headers=headers)
        antonyms = response.json()['antonyms']
        local_antonyms[word] = antonyms
        f.seek(0)
        f.write((json.dumps(local_antonyms)))
        f.truncate()
        if len(antonyms) is 0:
            return word
        return antonyms[0]



#FromAPI
def getRandomWordAPI() -> str:
    url = "https://wordsapiv1.p.rapidapi.com/words/"
    querystring = {"random":"true"}
    headers = {
        "X-RapidAPI-Key": "41b9c2ee17msh295225a18398362p1c732cjsn4afb01c0b61f",
        "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    return response.json()["word"]

#From json files
import random
def getRandomWordJSON() -> str:
    with open("./mutation_data/random_word.json", "r") as file:
        words = dict(json.load(file))['word']
        words : list
        return random.choice(words)

def getRandomVerbJSON() -> str:
    with open("./mutation_data/random_verbs.json", "r") as file:
        verbs = dict(json.load(file))['verb']
        verbs : list
        return random.choice(verbs)


def getRandomAdverbJSON() -> str:
    with open("./mutation_data/random_adverbs.json", "r") as file:
        adverbs = dict(json.load(file))['adverb']
        adverbs : list
        return random.choice(adverbs)

def getRandomAdjectiveJSON() -> str:
    with open("./mutation_data/random_adjectivea.json", "r") as file:
        adjectives = dict(json.load(file))['adjective']
        adjectives : list
        return random.choice(adjectives)

#TODO: Might be too large a set of misspellings
def getMisspellListJSON() -> dict:
    with open("./mutation_data/misspellings.json", "r") as file:
        misspellings = dict(json.load(file))
        actualMispells = {}
        for word, missSpells in misspellings.items():
            if len(missSpells) > 0:
                actualMispells[word] = random.choice(missSpells)
        return actualMispells

def populateRandomWord(count = 1):
    with open("./mutation_data/random_word.json", "r+") as f:
        local_words = json.load(f)
        if "word" not in local_words:
            local_words = {"word" : []}
        for i in range(0, count):
            word = getRandomWordAPI()
            if word in local_words["word"]:
                continue
            local_words["word"].append(word)
        f.seek(0)
        f.write((json.dumps(local_words)))
        f.truncate()