import json

jsonFile = open('captions_val2017.json')
jsonData = json.load(jsonFile)
outData = open('./captions/humanCOCO.txt', 'w')

captionDict = {}
for i, caption in enumerate(jsonData['annotations']):
    if(caption['image_id'] < 100_000_000):
        captionDict[caption['image_id']] = []
for i, caption in enumerate(jsonData['annotations']):
    if(caption['image_id'] < 100_000_000):
        captionDict[caption['image_id']].append(caption['caption'])
# print(captionDict)
captionkeys = []
for key in captionDict.keys():
    captionkeys.append(key)
captionkeys.sort()
for key in captionkeys:
    for sentence in captionDict[key]:
        sentence = sentence.replace("\n", "").replace(",", "").replace(".", "").replace("'", "").lower()
        outData.write(str(key) + ":" + str(sentence) + "\n")

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

with open("./captions/humanCOCO.txt", "r") as f:
    with open("./captions/HumanCOCOFullValidate.json", "w+") as g:
        json.dump(parseTxt1ByCaption(f.readlines(), False), g)