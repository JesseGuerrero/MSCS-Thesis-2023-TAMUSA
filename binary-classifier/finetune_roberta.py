# This is a illustration of how we might finetune the RoBERTa-based detector
# There will be syntax errors, but the logic should be correct.
import json
import numpy as np

import os
import torch
import tqdm
from torch import nn, optim
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from mutation_miniframework.Dataset import Dataset

roberta_detector_ckpt_dir = './'
roberta_detector_ckpt_name = 'detector-large.pt'
roberta_detector_ckpt_path = os.path.join(roberta_detector_ckpt_dir, roberta_detector_ckpt_name)

roberta_detector_ckpt_url = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download RoBERTa-based Detector ckpt if needed
if(not os.path.exists(roberta_detector_ckpt_path)):
  print("No RoBERTa!")
  exit(0)
# if (not os.path.exists(roberta_detector_ckpt_path)):
#   if(not os.path.exists(roberta_detector_ckpt_dir)):
#     os.makedirs(roberta_detector_ckpt_dir)
#   download_roberta_ckpt(ckpt_url, roberta_detector_ckpt_path)

# Initiate pre-trained RoBERTa-based Detector
ckpt = torch.load(roberta_detector_ckpt_path) # checkpoint for pre-trained reberta detector
model = RobertaForSequenceClassification.from_pretrained('roberta-large')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model : RobertaForSequenceClassification


# binary cross-entropy loss
BCE = nn.BCELoss()

# SGD optimizer (you may also try Adam and others),
# make sure to play with the learning rate to find one works
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# split the dataset into train, val, test (70% : 10%: 20%)
# use the train_captions to train the model

with open("../humanCOCOSmall.json", "r") as f:
  data = dict(json.load(f))
  dataset = Dataset(data, ["showcase"])

for epoch in range(10):

  ## train the model ##


  #model.train()

  # assume train_captions is lists of captions
  # random the train_captions set for each epoch
  # use np.random.shuffle


  for chunk in tqdm.tqdm(dataset.items()):
    # load K instances to form a train batch
    caption_batch_human = []
    caption_batch_machine = []
    mask_human = []
    mask_machine = []
    print(chunk[0])
    print(chunk[1])

    for i in range(100):
      img_name, chunk = (chunk[0], chunk[1])

      # form the human text batch
      tokens = tokenizer.encode(chunk)
      tokens = torch.Tensor(tokens)
      tokens = tokens.unsqueeze(0).long().to(device)
      mask = torch.ones_like(tokens).long().to(device)

      caption_batch_human.append([[tokens], [mask], 0]) # after the for-loop, we should have a K-by-3 list.
                                                        # 1st column: text embedding
                                                        # 2nd column: attention mask
                                                        # 3rd column: label for human or machine

      # form the machine text batch
      machine_caption = mutation(chunk) # random apply one mutation operator

      tokens = tokenizer.encode(machine_caption)
      tokens = torch.Tensor(tokens)
      tokens = tokens.unsqueeze(0).long().to(device)
      mask = torch.ones_like(tokens).long().to(device)
      caption_batch_machine.append([[tokens], 1])

    # concatenate two lists together
    caption_batch =  caption_batch_human + caption_batch_machine # the lenght of the list should be 2K-by-3
    np.random.shuffle(caption_batch) # we don't want all the human text together, and all machine text together

    # split the list to data, mask and label
    data = caption_batch[:,0]
    attention_mask = caption_batch[:,1]
    label = caption_batch[:,2]

    # train the model
    optimizer.zero_grad()
    logits = model(data, attention_mask=attention_mask)
    loss = BCE(logits, label) # gt_label: ground truth label, i.e., whether the text is human of machine
    loss.backward()
    optimizer.step()

    # get predicted result
    probs = logits[0].softmax(dim=-1)
    probs = probs.detach().cpu().flatten().numpy()
    result = np.argmax(probs)

    # use sklearn to calculate accuracy

    # might print accuracy periodically

  ## validate the model ##
  model.eval()
  with torch.no_grad():
    for chunk in tqdm.tqdm(val_captions):
      # load K instances to form a train batch
      for i in range(k):
        ...
      # concatenate two lists together
      # split the list to data, mask and label
      # val the model
      logits = model(val_data, attention_mask=attention_mask)
      # get predicted result
      probs = logits[0].softmax(dim=-1)
      probs = probs.detach().cpu().flatten().numpy()
      result = np.argmax(probs)

      # use sklearn to calculate accuracy

      # depending on the accuracy, you may want to save the ckpt
      # in-general, we want to use val to find the training ckpt that with the
      # highest val accuracy

# After the training is completed and the best model is selected based on Val Acc
# Test the model, ensentially the same with what you did before.
# The only difference is to use the finetuned CKPT instead of the RoBERTa provided one
