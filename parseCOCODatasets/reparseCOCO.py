import json
import random

# jsonFile = open('captions_val2017.json')
# jsonData = json.load(jsonFile)
# outData = open('./captions/humanCOCO.txt', 'w')
#
# captionDict = {}
# for i, caption in enumerate(jsonData['annotations']):
#     if(caption['image_id'] < 100_000_000):
#         captionDict[caption['image_id']] = []
# for i, caption in enumerate(jsonData['annotations']):
#     if(caption['image_id'] < 100_000_000):
#         captionDict[caption['image_id']].append(caption['caption'])
# # print(captionDict)
# captionkeys = []
# for key in captionDict.keys():
#     captionkeys.append(key)
# captionkeys.sort()
# for key in captionkeys:
#     for sentence in captionDict[key]:
#         sentence = sentence.replace("\n", "").replace(",", "").replace(".", "").replace("'", "").lower()
#         outData.write(str(key) + ":" + str(sentence) + "\n")

def parseTxt1ByCaption(lines : list, by_image : bool) -> dict:
    data = {}
    if by_image:
        for line in lines:
            if line.split(":")[0] in data:
                data[line.split(":")[0]][0] = (data[line.split(":")[0]][0] + line.split(":")[1].replace("\n", " "))
            if line.split(":")[0] not in data:
                data[line.split(":")[0]] = [line.split(":")[1].replace("\n", " ")]
    if not by_image:
        for line in lines:
            if line.split(":")[0] in data:
                data[line.split(":")[0]].append(line.split(":")[1].replace("\n", ""))
            if line.split(":")[0] not in data:
                data[line.split(":")[0]] = [line.split(":")[1].replace("\n", "")]
    return data

# with open("./captions/humanCOCO.txt", "r") as f:
#     with open("./captions/HumanCOCOFullTest.json", "w+") as g:
#         json.dump(parseTxt1ByCaption(f.readlines(), False), g)

# with open("./captions/HumanCOCOFullTest.json", "r+") as f:
#     data = dict(json.load(f))
#     caption_count = len(data.items())
#     captions = []
#     for img_name, caption in data.items():
#         captions.append((img_name, caption))
#     random.shuffle(captions)
#
#     validation_set = captions[:int(caption_count*0.5)]
#     training_set = captions[int(caption_count*0.5):]
#     with open("./captions/CompleteTestSet1.json", "w+") as file:
#         data = {}
#         for img_name, caption in validation_set:
#             try:
#                 if len(data[img_name]) is 0:
#                     data[img_name] = caption
#             except KeyError:
#                 data[img_name] = caption
#                 continue
#             data[img_name].append(caption)
#         json.dump(data, file)
#     with open("./captions/CompleteTestSet2.json", "w+") as file:
#         data = {}
#         for img_name, caption in training_set:
#             try:
#                 if len(data[img_name]) is 0:
#                     data[img_name] = caption
#             except KeyError:
#                 data[img_name] = caption
#                 continue
#             data[img_name].append(caption)
#         json.dump(data, file)


### for Creating test sets ###
# from mutation_miniframework.Dataset import *
# from mutation_miniframework.base_mutators import *
# from mutation_miniframework.operators import *
# import binary_classifier.utils as U
# misspellings = U.loadJSONWordDictionary("../mutation_miniframework/mutation_data/misspellings.json")
# antonyms = U.loadJSONWordDictionary("../mutation_miniframework/mutation_data/antonyms.json")
# synonyms = U.loadJSONWordDictionary("../mutation_miniframework/mutation_data/misspellings.json")
# randomList = []
# with open("../mutation_miniframework/mutation_data/random_word.json") as randomJSON:
#     randomBuffer = dict(json.load(randomJSON))
#     randomList = randomBuffer["word"]
#
# with open("./captions/CompleteValidationSet.json", "r") as f:
#     data = dict(json.load(f))
#     dataDict={}
#     resultMutations = {}
#     for img_name, captions in data.items():
#         print(img_name)
#         for caption in captions:
#             dataDict[0] = [caption]
#             choice = random.randrange(0, 6)
#             mutatedCaptionData = Dataset(dataDict, [""])
#             if choice == 0:
#                 deleteRandomArticle(mutatedCaptionData, [" a ", " an ", " the ", " is "], "", word_change_limit=3)
#             if choice == 1:
#                 replaceLetters(mutatedCaptionData, {
#                     "a": "α",
#                     "e": "ε"
#                 }, "", word_change_limit=3)
#             if choice == 2:
#                 replaceFromDictionary(mutatedCaptionData, misspellings, "", word_change_limit=3)
#             if choice == 3:#random word replacement
#                 replaceWordListWithRandomSelf(mutatedCaptionData, randomList, "", word_change_limit=2)
#             if choice == 4:#synonyms replacement
#                 replaceFromDictionary(mutatedCaptionData, synonyms, "", word_change_limit=2)
#             if choice == 5:#antonyms replacement
#                 replaceFromDictionary(mutatedCaptionData, antonyms, "", word_change_limit=2)
#             if choice == 6:
#                 pass
#
#             try:
#                 if len(resultMutations[img_name]) is 0:
#                     resultMutations[img_name] = [mutatedCaptionData[0][0]]
#             except KeyError:
#                 resultMutations[img_name] = [mutatedCaptionData[0][0]]
#                 continue
#             resultMutations[img_name].append(mutatedCaptionData[0][0])
#     with open("./captions/MutationCompleteValidationSet1.json", "w+") as file:
#         json.dump(resultMutations, file)

from mutation_miniframework.Dataset import *
from mutation_miniframework.base_mutators import *
from mutation_miniframework.operators import *
with open("./captions/MutationFullSet_val.json", "r") as file:
    data = dict(json.load(file))
    newSize = int(len(data.items())/4)
    captions = []
    for img_name, caption in data.items():
        captions.append((img_name, caption))

    new_set = captions[:newSize]
    with open("./captions/MutationQuarterSet_val.json", "w+") as file:
        data = {}
        for img_name, caption in new_set:
            try:
                if len(data[img_name]) is 0:
                    data[img_name] = caption
            except KeyError:
                data[img_name] = caption
                continue
            data[img_name].append(caption)
        json.dump(data, file)