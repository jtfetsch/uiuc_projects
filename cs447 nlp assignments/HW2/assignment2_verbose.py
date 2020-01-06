# -*- coding: utf-8 -*-
"""Assignment2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15PslaHjvVQp2wClc062GaovhVUUtY_R4

# Assignment 2

In this part of assignment 2 we'll be building a machine learning model to detect sentiment of movie reviews using the Stanford Sentiment Treebank([SST])(http://ai.stanford.edu/~amaas/data/sentiment/) dataset. First we will import all the required libraries. We highly recommend that you finish the PyTorch Tutorials [ 1 ](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html),[ 2 ](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html),[ 3 ](https://github.com/yunjey/pytorch-tutorial). before starting this assignment. After finishing this assignment we will able to answer the following questions-


* How to write Dataloaders in Pytorch?
* How to build dictionaries and vocabularies for Deep Nets?
* How to use Embedding Layers in Pytorch?
* How to build various recurrent models (LSTMs and GRUs) for sentiment analysis?
* How to use packed_padded_sequences for sequential models?

# Import Libraries
"""

import numpy as np

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict
from torchtext import datasets
from torchtext import data

from torch.nn.utils.rnn import pack_sequence, pad_sequence

"""## Download dataset
First we will download the dataset using [torchtext](https://torchtext.readthedocs.io/en/latest/index.html), which is a package that supports NLP for PyTorch. The following command will get you 3 objects `train_data`, `val_data` and `test_data`. To access the data:

*   To access list of textual tokens - `train_data[0].text`
*   To access label - `train_data[0].label`
"""

if(__name__=='__main__'):
  train_data, val_data, test_data = datasets.SST.splits(data.Field(tokenize = 'spacy'), data.LabelField(dtype = torch.float), filter_pred=lambda ex: ex.label != 'neutral')

if(__name__=='__main__'):
  print(train_data[0].text)
  print(train_data[0].label)

"""## Define the Dataset Class

In the following cell, we will define the dataset class. You need to implement the following functions: 


*   ` build_dictionary() ` - creates the dictionaries `ixtoword` and `wordtoix`. Converts all the text of all examples, in the form of text ids and stores them in `textual_ids`. If a word is not present in your dictionary, it should use `<unk>`. Use the hyperparameter `THRESHOLD` to control the words to be in the dictionary based on their occurrence. Note the occurrences should be `>=THRESHOLD` to be included in the dictionary.
*   ` get_label() ` - It should return the value `0` if the label in the dataset is `positive`, and should return `1` if it is `negative`. 
*   ` get_text() ` - This function should pad the review with `<end>` character uptil a length of `MAX_LEN` if the length of the text is less than the `MAX_LEN`.
*   ` __len__() ` - This function should return the total length of the dataset.
*   ` __getitem__() ` - This function should return the padded text, the length of the text (without the padding) and the label.
"""

THRESHOLD = 10
MAX_LEN = 60
INPUT_DIM = 17  # needed for autograder
BATCH_SIZE = 32
UNK = 1
END = 0
class TextDataset(data.Dataset):
  def __init__(self, examples, split, ixtoword=None, wordtoix=None, THRESHOLD=THRESHOLD):
    print("starting load")
    global INPUT_DIM
    self.examples = examples
    self.split = split
    self.THRESHOLD = THRESHOLD
    self.textual_ids, self.textual_labels, self.ixtoword, self.wordtoix = self.build_dictionary(wordtoix, ixtoword)
    print("Built dicts with len =",INPUT_DIM)
    print("  from source with",len(self.examples),"reviews")
    print("  producing",len(self.textual_ids),"index vectors")
    print("  with",self.positives,"positive reviews and",len(self.textual_ids)-self.positives,"negative ones")
    print("  at Threshold =",THRESHOLD)

  def build_dictionary(self, wordtoix1=None, ixtoword1=None):
    print("starting build dictionary")
    global INPUT_DIM
    # do count checks for threshold comparisons
    wordCounts = {}
    for example in self.examples:
      for token in example.text:
        token = token.lower()
        if token not in wordCounts:
          wordCounts[token] = 1
        else:
          wordCounts[token] += 1
    ixtoword=ixtoword1
    wordtoix=wordtoix1

    textual_ids = []
    textual_labels = []

    if wordtoix==None and ixtoword==None:
      # create indices for all entries with count >= Threshold
      ixtoword = {}
      wordtoix = {}
      nextIndex = 2

      ### <end> should be at idx 0
      ### <unk> should be at idx 1
      ixtoword[END] = "<end>"
      wordtoix["<end>"] = END
      ixtoword[UNK] = "<unk>"
      wordtoix["<unk>"] = UNK

      # index mappings complete: usable to determine <unk>
      for word in wordCounts.keys():
        if wordCounts[word] >= self.THRESHOLD:
          ixtoword[nextIndex] = word
          wordtoix[word] = nextIndex
          nextIndex += 1

    self.positives = 0

    # create textual_ids
    for index in range(len(self.examples)):
      sentence = []
      for token in self.examples[index].text:
        if token in wordtoix.keys():
          sentence.append(wordtoix[token])
        else:
          sentence.append(UNK)

      textual_ids.append(sentence)
      int_label = 0 if self.examples[index].label == 'positive' else 1
      textual_labels.append(int_label)
      self.positives += 1-int_label

    INPUT_DIM = len(ixtoword.keys())

    return textual_ids, textual_labels, ixtoword, wordtoix

  def get_label(self, index):
    print("get_label",index)
    return torch.Tensor([self.textual_labels[index]])

  def get_text(self, index):
    print("get_text",index)
    sentence = self.textual_ids[index][:MAX_LEN]
    while len(sentence) < MAX_LEN:
      sentence.append(END)
    print("get_text_end",index)
    return torch.Tensor(sentence).float()

  def __len__(self):
    print("__len__")
    return len(self.textual_ids)

  def __getitem__(self, index):
    print("__getitem__", index)
    text = self.get_text(index)
    text_len = len(self.textual_ids[index])
    lbl = self.get_label(index)
    print("__getitem__ return", text, text_len, lbl)
    return text, text_len, lbl

"""## Initialize the Dataloader
We initialize the training and testing dataloaders using the Dataset classes we create for both training and testing. Make sure you use the same vocabulary for both the datasets.
"""

if(__name__=='__main__'):
  print("main loader")
  Ds = TextDataset(train_data, 'train')
  train_loader = torch.utils.data.DataLoader(Ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
  test_Ds = TextDataset(test_data, 'test', wordtoix=Ds.wordtoix, ixtoword=Ds.ixtoword)
  test_loader = torch.utils.data.DataLoader(test_Ds, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

"""## Build your Sequential Model
In the following we provide you the class to build your model. We provide some parameters, we expect you to use in the initialization of your sequential model.
"""

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        print("RNN start")
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.pad_idx = pad_idx
        self.bidirectional = bidirectional
        self.embeds = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = n_layers, dropout = dropout, bidirectional = bidirectional)
        self.linear = nn.Linear(hidden_dim*(2 if bidirectional else 1), output_dim)
        self.hidden = None
        self.c = None

    def init_hidden(self, batch_size):
        print("init_hidden")
        self.hidden = torch.rand(self.n_layers*(2 if self.bidirectional else 1), batch_size, self.hidden_dim).float()
        self.c = torch.rand(self.n_layers*(2 if self.bidirectional else 1), batch_size, self.hidden_dim).float()

    def forward(self, text, text_lengths):
        print("forward")
        # INPUTS
        #text = [MAX LEN, batch size]
        #text_lengths = [batch size]
        embeds = self.embeds(text.long())
        #embeds = [MAX_LEN, batch_size, embedding_dim]
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, text_lengths, enforce_sorted=False).float()
        embeds.to(device)

        # LSTM
        if self.hidden is None:
          self.init_hidden(len(text_lengths))
        lstm_out, (self.hidden, self.c) = self.rnn(embeds, (self.hidden, self.c))
        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out, total_length=MAX_LEN)
        #hidden = [n_layers*num_directions, batch_size, hidden_dim]
        #c = [n_layers*num_directions, batch_size, hidden_dim]
        #lstm_out[0] = [MAX_LEN, batch_size, hidden_dim*num_directions]
        #lstm_out[1] = text_lengths


        # LINEAR, OUTPUT
        linear_input = lstm_out[0][0]
        y_pred = self.linear(linear_input)
        #y_pred = [batch_size,1]
        return y_pred

# Hyperparameters for your model
# Feel Free to play around with these
# for getting optimal performance

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.6
PAD_IDX = 0
LEARNING_RATE = 2.5e-3

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)


def count_parameters(model):
    print("count_params")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if(__name__=='__main__'):
  print('The model has {count_parameters(model):,} trainable parameters')

"""### Put your model on the GPU

### Define your loss function and optimizer
"""

if(__name__=='__main__'):
  print("cuda")
  model = model.to(device)

# Play around with different optimizers and loss functions
# for getting optimal performance
# For optimizers : https://pytorch.org/docs/stable/optim.html
# For loss functions : https://pytorch.org/docs/stable/nn.html#loss-functions
if(__name__=='__main__'):
  #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  #optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
  criterion = nn.L1Loss()
  criterion.to(device)

def binary_accuracy(preds, y):
    print("binary")
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

"""## Train your Model"""

import datetime
def train_model(model, num_epochs, data_loader):
  print("train")
  start = datetime.datetime.now()
  model.train()
  for epoch in range(10):
    epoch_loss = 0
    epoch_acc = 0
    for idx, (text, text_lens, label) in enumerate(data_loader):
        if(idx%100==0):
          print('Executed Step {} of Epoch {}'.format(idx, epoch))
        text = text.to(device)
        # text - [batch_len, MAX_LEN]
        text_lens = text_lens.to(device)
        # text - [batch_len]
        label = label.float()
        label = label.to(device)

        optimizer.zero_grad()
        text = text.permute(1, 0) # permute for sentence_len first for embedding
        predictions = model(text, text_lens)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # allow backprop of loss calculations
        model.hidden.detach_()
        model.c.detach_()
        model.hidden = model.hidden.detach()
        model.c = model.c.detach()
        model.to(device)
    print('Training Loss Value of Epoch {} = {}'.format(epoch ,epoch_loss/len(train_loader)))
    print('Training Accuracy of Epoch {} = {}'.format(epoch ,epoch_acc/len(train_loader)))

    # if we are stuck with terrible accuracy, refresh the hidden layers
    #if epoch_acc/len(train_loader) < 0.54:
    #  global BATCH_SIZE
    #  print("Resetting hidden layers due to low accuracy")
    #  model.init_hidden(BATCH_SIZE)
    #  model.to(device)

    end = datetime.datetime.now()
    print(end-start)

"""## Evaluate your Model"""

def evaluate(model, data_loader):
  print("evaluate")
  model.eval()
  epoch_loss = 0
  epoch_acc = 0
  all_predictions = []
  for idx, (text, text_lens, label) in enumerate(data_loader):
      if(idx%100==0):
        print('Executed Step {}'.format(idx))
      text = text.permute(1, 0)
      text = text.to(device)
      text_lens = text_lens.to(device)
      label = label.float()
      label = label.to(device)
      optimizer.zero_grad()

      predictions = model(text, text_lens)
      all_predictions.append(torch.round(torch.sigmoid(predictions)))
      loss = criterion(predictions, label)
      acc = binary_accuracy(predictions, label)
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  print(epoch_loss/len(data_loader))
  print(epoch_acc/len(data_loader))
  predictions = torch.cat(all_predictions)
  return predictions

"""## Training and Evaluation

We first train your model using the training data. Feel free to play around with the number of epochs. We recommend **you write code to save your model** [(save/load model tutorial)](https://pytorch.org/tutorials/beginner/saving_loading_models.html) as colab connections are not permanent and it can get messy if you'll have to train your model again and again.
"""

if(__name__=='__main__'):
  print("colab")
  try:
    from google.colab import drive
    drive.mount('/content/drive')
  except:
    pass
  #model = torch.load('drive/My Drive/UIUC Classes/CS 447 Natural Language Processing/Assignments/HW2/model')
  train_model(model, 10, train_loader)
  torch.save(model, 'drive/My Drive/UIUC Classes/CS 447 Natural Language Processing/Assignments/HW2/model')

"""Now we will evaluate your model on the test set."""

if(__name__=='__main__'):
  print("actually evaluate")
  predictions = evaluate(model, test_loader)
  predictions = predictions.cpu().data.detach().numpy()
  assert(len(predictions)==len(test_data))

"""## Saving results for Submission
Saving your test results for submission. You will save the `result.txt` with your test data results. Make sure you do not **shuffle** the order of the `test_data` or the autograder will give you a bad score.

You will submit the following files to the autograder on the gradescope :


1.   Your `result.txt` of test data results
2.   Your code of this notebook. You can do it by clicking `File`-> `Download .py` - make sure the name of the downloaded file is `assignment2.py`
"""

if(__name__=='__main__'):
  print("save")
  try:
    from google.colab import drive
    drive.mount('/content/drive')
  except:
    pass
  np.savetxt('drive/My Drive/UIUC Classes/CS 447 Natural Language Processing/Assignments/HW2/result.txt', predictions, delimiter=',')