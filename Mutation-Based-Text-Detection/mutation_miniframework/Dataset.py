import json, os

class Dataset:
    '''
    Wraps the Python Dictionary built-in class with some extra functions
    -Saves original dataset for reverting
    -Saves a root directory for the dataset for re-use
    -Unique file name saving per dataset
    Dictionary format
    {
        Unique_ID: [TEXT1, TEXT2, ETC],
        Unique_ID: [TEXT1, TEXT2, ETC],
        Unique_ID: [ETC]
    }
    '''
    _data : dict
    _original_data : dict
    _root_dir = "."
    # List of strings to add to file name
    _mutations: list

    def __init__(self, data: dict, mutations: list):
        self._data = data
        self._original_data = data
        self._mutations = mutations

    def resetData(self):
        self._data = self._original_data
        self._mutations = []

    def saveDataMutation(self, mutation_append: list):
        self._mutations.append(mutation_append)

    def _determineFileName(self) -> str:
        base_name = ""
        for mutation in self._mutations:
            base_name = base_name + mutation
        file_name = base_name + ".json"
        files_in_directory = os.listdir(self._root_dir)
        for i in range(1, 10_000):
            if (file_name not in files_in_directory):
                break
            if (file_name in files_in_directory):
                file_name = base_name + str(i) + ".json"
        return self._root_dir + file_name

    def setRootDir(self, root: str):
        self._root_dir = root

    def saveToFile(self):
        with open(self._determineFileName(), "w") as outfile:
            json.dump(self._data, outfile)

    #Behave as a dictionary
    def keys(self) -> list:
        return self._data.keys()

    def values(self) -> list:
        return self._data.values()

    def items(self) -> tuple:
        return self._data.items()

    #Operator overloads
    def __str__(self):
        return str(self._data)
    def __setitem__(self, key, value):
        self._data[key] = value
    def __getitem__(self, key):
        return self._data[key]


'''
Purely for reformatting the previous dataset.
'''
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