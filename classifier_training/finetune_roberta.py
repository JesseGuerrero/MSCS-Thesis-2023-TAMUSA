# The pre-trained roberta-based detector may only works w/
# - Huggingface version 2.9.1 (i.e., ```transformers==2.9.1```)
# - ```tokenizers==0.7.0```
# !pip install transformers==2.9.1

'''
~~~About Checkpoints~~~
base.pt is the most accurate checkpoint
base_1.pt is the latest checkpoint
'''
import numpy

FROM_CHECKPOINT = False
CHECKPOINTNAME = ""

import os
import csv
import time
import tqdm
# from tqdm.notebook import trange
from tqdm import trange
import math
import numpy as np
import torch
from torch import nn, optim
from transformers import RobertaForSequenceClassification, RobertaTokenizer

import utils as U

import sys

# setting path
sys.path.append('..')

from mutation_miniframework.Dataset import *
from mutation_miniframework.operators import deleteRandomArticle, replaceLetters, replaceFromDictionary, \
	replaceWordListWithRandomSelf

misspellings = U.loadJSONWordDictionary("../mutation_miniframework/mutation_data/misspellings.json")
antonyms = U.loadJSONWordDictionary("../mutation_miniframework/mutation_data/antonyms.json")
synonyms = U.loadJSONWordDictionary("../mutation_miniframework/mutation_data/misspellings.json")
randomList = []
with open("../mutation_miniframework/mutation_data/random_word.json") as randomJSON:
	randomBuffer = dict(json.load(randomJSON))
	randomList = randomBuffer["word"]

project_data_path = "./test"

text_data_path = os.path.join(project_data_path, 'data_10k', 'Parsed')
real_text_dir = os.path.join(text_data_path, 'train_val_test/real')
fake_text_dir = os.path.join(text_data_path, 'train_val_test/mutation')
text_file_mutation = 'MutationFullSet.json'
text_file_real = 'HumanFullSet.json'


ckpt_dir = os.path.join(project_data_path, "ckpt")
output_path = os.path.join(ckpt_dir, "COCO-Finetune-RoBERTa-Based-Detector")
if(not os.path.exists(output_path)):
	print("Making Dir...\n\t%s" %output_path)
	os.makedirs(output_path)

roberta_detector_ckpt_dir = os.path.join(ckpt_dir, 'RoBERTa-Based-Detector')
roberta_detector_ckpt_name = 'detector-large.pt'
roberta_detector_ckpt_path = os.path.join(roberta_detector_ckpt_dir,
										  roberta_detector_ckpt_name)
roberta_detector_ckpt_url = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt'

# Download RoBERTa-based Detector ckpt if needed
if (not os.path.exists(roberta_detector_ckpt_path)):
	if(not os.path.exists(roberta_detector_ckpt_dir)):
		print("Making Dir...\n\t%s" %roberta_detector_ckpt_dir)
		os.makedirs(roberta_detector_ckpt_dir)
	U.download_roberta_ckpt(roberta_detector_ckpt_url,
							roberta_detector_ckpt_path)

# Load data
#[img name, captions, label]
train_data = U.load_data(real_text_dir, text_file_real,
						 fake_text_dir, text_file_mutation,
						 train_test_split='train')
val_data = U.load_data(real_text_dir, text_file_real,
					   fake_text_dir, text_file_mutation,
					   train_test_split='val')
test_data = U.load_data(real_text_dir, text_file_real,
						fake_text_dir, text_file_mutation,
						train_test_split='test')

# set hyperparameters
batch_size = 1
epochs = 10
learning_rate = 0.0001
finetune_embeddings = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initiate pre-trained RoBERTa-based Detector
ckpt = None
if FROM_CHECKPOINT is True:
	ckpt = torch.load(os.path.join(output_path, CHECKPOINTNAME))
else:
	ckpt = torch.load(roberta_detector_ckpt_path) # checkpoint for pre-trained reberta detector, replace here with path.
model = RobertaForSequenceClassification.from_pretrained('roberta-large')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

if FROM_CHECKPOINT is True:
	model.load_state_dict(ckpt)#Only do ckpt when loading
else:
	model.load_state_dict(ckpt['model_state_dict'])#Only do ckpt when loading
model = model.to(device)

# Freeze roberta weights (i.e., the embedding weights)
# leave the classifier tunable
for p in model.roberta.parameters():
	p.requires_grad = finetune_embeddings

BCE = nn.BCELoss()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

## Train Loop ##
t = trange(epochs, desc="", position=0, leave=True)

phases = ["train", "val"]
best_val_acc = 0
best_epoch = 0
train_hist = []
val_hist = []

from random import randrange
for e in t:
	model.to(device)
	for phase in phases:

		## Initialization ##
		if phase == "train":
			model.train()
			data = np.array(train_data)
			np.random.shuffle(data)
		else:
			model.eval()
			data = np.array(val_data)

		epoch_loss = 0
		epoch_correct_pred = 0
		step_per_epoch = math.floor(len(data) / batch_size)

		## Train/Val Loop ##
		for i in range(step_per_epoch):
			# Load one batch of data
			# batch size might have to be 1 due to varying caption length
			cur_data = data[i*batch_size:(i+1)*batch_size]
			cur_names = cur_data[:,0]
			cur_captions = cur_data[:,1]#TODO: Adjust for extra dimension, captions and labels
			cur_labels = cur_data[:,2].astype(np.int8)#Mutation is 0, real is 1

			# Generate mutation
			# if need to generate mutation, add code here
			# print("DATA " + str(cur_data))
			dataDict = {}
			if(phase == "train" and cur_labels == 0):
				dataDict[0] = [str(cur_data[:,1])]
				choice = randrange(0, 6)
				mutatedCaptionData = Dataset(dataDict, [""])
				if choice == 0:
					deleteRandomArticle(mutatedCaptionData, [" a ", " an ", " the "], "", word_change_limit=99)
				if choice == 1:
					replaceLetters(mutatedCaptionData, {
						"a": "α",
						"e": "ε"
					}, "", word_change_limit=3)
				if choice == 2:
					replaceFromDictionary(mutatedCaptionData, misspellings, "", word_change_limit=3)
				if choice == 3:#random word replacement
					replaceWordListWithRandomSelf(mutatedCaptionData, randomList, "", word_change_limit=2)
				if choice == 4:#synonyms replacement
					replaceFromDictionary(mutatedCaptionData, synonyms, "", word_change_limit=2)
				if choice == 5:#antonyms replacement
					replaceFromDictionary(mutatedCaptionData, antonyms, "", word_change_limit=2)
				if choice == 6:
					pass
				# print("MUTATION " + mutatedCaptionData[0][0])
				cur_data[:,1] = mutatedCaptionData[0][0]
				cur_captions = cur_data[:,1]
				# print(str(choice) + ":" + str(cur_data[:,1][0]))

			# Tokenize captions
			cur_token_ids = [tokenizer.encode(item) for item in cur_captions]
			cur_masks = [np.ones(len(item)) for item in cur_token_ids]

			# Convert to tensor and send data to device
			cur_token_ids = torch.tensor(np.array(cur_token_ids)).to(device)
			cur_labels = torch.tensor(np.array(cur_labels)).long().to(device)
			cur_masks = torch.tensor(np.array(cur_masks)).to(device)

			# For training
			if(phase == "train"):
				optimizer.zero_grad()
				logits = model(cur_token_ids, attention_mask=cur_masks)
				loss = loss_function(logits[0], cur_labels)
				loss.backward()
				optimizer.step()
				# scheduler may be needed in the future

			# For validation
			else:
				with torch.no_grad():
					logits = model(cur_token_ids, attention_mask=cur_masks)
					loss = loss_function(logits[0], cur_labels)

			# Track current performance
			# Count correct prediciton
			for kk in range(len(cur_labels)):
				prob = logits[0][kk].softmax(dim=-1)
				pred = torch.argmax(prob.detach().cpu())
				if(pred==cur_labels[kk]):
					epoch_correct_pred +=  1.0
			# Add current loss to total epoch loss
			epoch_loss += loss.item()

			# Update progress bar
			t.set_description("Epoch/Step: %i/%i[Phase:%s]  Loss:%.4f  CorrectPred:%.4f [%i/%i]"
				% (e, i, phase, loss.item(), epoch_correct_pred/(i+1), epoch_correct_pred, (i+1)))


		## Compute epoch performance ##
		epoch_acc = epoch_correct_pred / (step_per_epoch * batch_size)
		epoch_loss = epoch_loss / step_per_epoch

		if(phase=="train"):
			train_hist.append([epoch_loss, epoch_acc])
			np.save(os.path.join(output_path,"train_hist.npy"),
					np.asarray(train_hist))

		else:
			val_hist.append([epoch_loss, epoch_acc])
			np.save(os.path.join(output_path,"val_hist.npy"),
					np.asarray(val_hist))

		if(phase == "val"):
			if(epoch_acc>best_val_acc):
				best_val_acc = epoch_acc
				best_epoch = e
				print("Epoch:%d Acc:%.4f higher than the previous best performance"
					  %(e, best_val_acc))
				print("Saving ckpt...")
				# save the CKPT
				torch.save(model.cpu().state_dict(),
						   os.path.join(output_path,"base.pt"))

	print("\nEpoch:%d   Train Loss/Acc: %.4f/%.4f   Val Loss/ACC %.4f/%.4f"
		  %(e, train_hist[e][0], train_hist[e][1], val_hist[e][0], val_hist[e][1]))

torch.save(model.cpu().state_dict(),
		   os.path.join(output_path,"base_"+str(e)+".pt"))
