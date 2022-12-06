import json

import time

import urllib.request
import progressbar
import os
import numpy as np
from tqdm import tqdm

def download_roberta_ckpt(ckpt_url, output_path, 
						  show_progress=True):
	print("Downloading RoBERTa-based detector ckpt...")
	time.sleep(0.1)
	class MyProgressBar():
		def __init__(self):
			self.pbar = None
		def __call__(self, block_num, block_size, total_size):
			if not self.pbar:
				self.pbar=progressbar.ProgressBar(maxval=total_size)
				self.pbar.start()
			downloaded = block_num * block_size
			if downloaded < total_size:
				self.pbar.update(downloaded)
			else:
				self.pbar.finish()
	if(show_progress):
		urllib.request.urlretrieve(ckpt_url, output_path, MyProgressBar())
	else:
		urllib.request.urlretrieve(ckpt_url, output_path)

def loadJSONWordDictionary(path : str) -> dict:
	wordDictionary = {}
	with open(path) as fileJSON:
		jsonDictionary = dict(json.load(fileJSON))
	for word, wordList in jsonDictionary.items():
		wordList : list
		if len(wordList) > 0:
			wordDictionary[word] = wordList[0]
	return wordDictionary


def load_text(cur_text_path):
	# this function load the captions from a .txt file
	# cur_text_path: the path of the .txt file
	captions = []
	img_names = []
	if ".json" in cur_text_path:
		with open(cur_text_path) as jsonFile:
			data = dict(json.load(jsonFile))
			for img, captionList in data.items():
				img_names.append(img)
				captionBuff = []
				for caption in captionList:
					captionBuff.append(caption)
				captions.append(" ".join(captionBuff))
	if ".txt" in cur_text_path:
		with open(cur_text_path) as txtFile:
			for line in txtFile.readlines():
				img_name, caption = parse_caption(line)
				img_names.append(img_name)
				captions.append(caption)
	return np.array(img_names), np.array(captions)

def parse_caption(caption : list):
	# This function takes an input like "34:a zebra standing on top of a lush green field \n"
	# This function extracts the image_name (e.g., 34) and the caption (e.g., the rest of the text)
	index = caption.index(':') # index of the 1st :
	img_name = caption[:index]
	caption = caption[index+1:]
	caption = caption.replace("\n", "") #remove newline character
	caption = caption.rstrip() #remove whitespace from the end
	return img_name, caption

# generate mutation for a single caption
def letter_mutation(caption, keyword="a",
					oletter="a", mletter="α",
					max_count = 10000):
	# This function will search for a keyword in a string and
	# replace certain letter with its mutation form
	# Inputs:
	#       caption: the input string
	#       keyword: the word where letter-level mutation happens
	#       oletter: the original letter in the keyword
	#       mletter: mutation letter, used to replace the original letter
	#       max_count: maximum number of letter changes for a string

	# seperate image_name and caption from the string
	img_name, s = parse_caption(caption)
	words = s.split(" ") # get a list of words
	count = 0 # used to count how many letter has been changed
	new_s = ""
	# find the keyword and do the mutation
	for word in words:
		if(count==max_count):
			break
		if(word==keyword):
			word = word.replace(oletter, mletter)
			count += 1
		new_s += word+" "
	return img_name+":"+new_s+"\n", count

def word_mutation(caption, keyword="a",
				  mword="α", max_count = 10000):
	# seperate image_name and caption from the string
	img_name, s = parse_caption(caption)
	words = s.split(" ") # get a list of words
	count = 0 # used to count how many letter has been changed
	new_s = ""
	# find the keyword and do the mutation
	for word in words:
		if(count==max_count):
			break
		if(word==keyword):
			word = mword
			count += 1
		new_s += word+" "
	return img_name+":"+new_s+"\n", count

# generate mutation set for multiple captions
def generate_letter_level_mutation(captions, keywords,
								   oletters, mletters,
								   max_count=10000):
	mcaptions = []
	ocaptions = []
	acaptions = []
	# print(keywords)
	for caption in tqdm.tqdm(captions):
		num_changes = 0
		for keyword in (keywords):
			for i in range(len(oletters)):
				if(num_changes>0):
					mcaption, count = letter_mutation(mcaption, 
													  keyword=keyword,
													  oletter=oletters[i], 
													  mletter=mletters[i],
													  max_count=max_count)
				else:
					mcaption, count = letter_mutation(caption, 
													  keyword=keyword,
													  oletter=oletters[i], 
													  mletter=mletters[i],
													  max_count=max_count)
				num_changes+=count

		if(num_changes>0):
			mcaptions.append(mcaption)
			ocaptions.append(caption)
		acaptions.append(mcaption)

	return mcaptions, ocaptions, acaptions

def generate_word_level_mutation(captions, keywords, 
								 mword="", max_count=10000):
	# Get word-level mutations
	mcaptions = []
	ocaptions = []
	acaptions = []
	for caption in tqdm.tqdm(captions):
		num_changes = 0
		for keyword in keywords:
			if(num_changes>0):
				mcaption, count = word_mutation(mcaption, 
												keyword=keyword, 
												mword=mword, 
												max_count=max_count)
			else:
				mcaption, count = word_mutation(caption, 
												keyword=keyword, 
												mword=mword, 
												max_count = max_count)
			num_changes+=count

		if(num_changes>0):
			mcaptions.append(mcaption)
			ocaptions.append(caption)
		acaptions.append(mcaption) # The list has the same length with the 
								   # original caption list. If there is no 
								   # mutation changes in an instance, add 
								   # the original to the list. If there are 
								   # mutation changes, add the mutated 
								   # instance to the list.

	return mcaptions, ocaptions, acaptions

# load real/fake text
def load_data(real_text_dir, text_file_real,
						fake_text_dir, text_file_fake,
						train_test_split='train'):
		# load real text
		cur_text_path = os.path.join(real_text_dir, 
									 train_test_split,
									 (train_test_split.title() + "_" + text_file_real))
		real_names, real_captions = load_text(cur_text_path)
		real_data = []
		for i in range(len(real_captions)):
			real_data.append([real_names[i], real_captions[i], 1])

		# load fake (machine-generated) text
		cur_text_path = os.path.join(fake_text_dir, 
									 train_test_split,
									 (train_test_split.title() + "_" + text_file_fake))
		fake_names, fake_captions = load_text(cur_text_path)
		fake_data = []
		for i in range(len(fake_captions)):
			real_data.append([fake_names[i], fake_captions[i], 0])

		# combine real/fake lists together and return
		return real_data+fake_data
