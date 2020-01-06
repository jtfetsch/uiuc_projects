# -*- coding: utf-8 -*-
"""Copy of MP4_bak.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16iR3ejgMEitkmxnb1LypzFOYZoM5dmVh

# Using Attention for Neural Machine Translation
In this notebook we are going to perform machine translation using a deep learning based approach and attention mechanism.

Specifically, we are going to train a sequence to sequence model for Spanish to English translation.  We will use Sequence to Sequence Models for this Assignment. In this assignment you only need tto implement the encoder and decoder, we implement all the data loading for you.Please **refer** to the following resources for more details:

1.   https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
2.   https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
3. https://arxiv.org/pdf/1409.0473.pdf
"""

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
print(torch.__version__)

"""# Download The Data

Here we will download the translation data. We will learn a model to translate Spanish to English.
"""

from google.colab import drive
drive.mount('/gdrive')

cd sample_data/

!wget http://www.manythings.org/anki/spa-eng.zip

!unzip spa-eng.zip

f = open('spa.txt', encoding='UTF-8').read().strip().split('\n')
lines = f
total_num_examples = 30000 
original_word_pairs = [[w for w in l.split('\t')][:2] for l in lines[:total_num_examples]]
data = pd.DataFrame(original_word_pairs, columns=["eng", "es"])
data # visualizing the data

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Preprocessing the sentence to add the start, end tokens and make them lower-case
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

# Now we do the preprocessing using pandas and lambdas
# Make sure YOU only run this once - if you run it twice it will mess up the data so you will have run the few above cells again
data["eng"] = data.eng.apply(lambda w: preprocess_sentence(w))
data["es"] = data.es.apply(lambda w: preprocess_sentence(w))
data[250:260]

"""# Vocabulary Class

We create a class here for managing our vocabulary as we did in MP2. In this MP, we have a separate class for the vocabulary as we need 2 different vocabularies - one for English and one for Spanish.
"""

class Vocab_Lang():
    def __init__(self, data):
        """ data is the list of all sentences in the language dataset"""
        self.data = data
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        
        self.create_index()
        
    def create_index(self):
        for sentence in self.data:
            # update with individual tokens
            self.vocab.update(sentence.split(' '))

        # add a padding token
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

# index language using the class above
inp_lang = Vocab_Lang(data["es"].values.tolist())
targ_lang = Vocab_Lang(data["eng"].values.tolist())
# Vectorize the input and target languages
input_tensor = [[inp_lang.word2idx[s] for s in es.split(' ')]  for es in data["es"].values.tolist()]
target_tensor = [[targ_lang.word2idx[s] for s in eng.split(' ')]  for eng in data["eng"].values.tolist()]

def max_length(tensor):
    return max(len(t) for t in tensor)

# calculate the max_length of input and output tensor for padding
max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

# pad all the sentences in the dataset with the max_length
input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]

# Creating training and test/val sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = input_tensor[:24000], input_tensor[24000:], target_tensor[:24000], target_tensor[24000:]

assert(len(input_tensor_train)==24000)
assert(len(target_tensor_train)==24000)
assert(len(input_tensor_val)==6000)
assert(len(target_tensor_val)==6000)

"""# Dataloader for our Encoder and Decoder

We prepare the dataloader and make sure the dataloader returns the source sentence, target sentence and the length of the source sentenc sampled from the training dataset.
"""

# conver the data to tensors and pass to the Dataloader 
# to create an batch iterator
from torch.utils.data import Dataset, DataLoader
class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # TODO: convert this into torch code is possible
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x,y,x_len
    
    def __len__(self):
        return len(self.data)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 60
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

train_dataset = MyData(input_tensor_train, target_tensor_train)
val_dataset = MyData(input_tensor_val, target_tensor_val)

dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)

val_dataset = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=False)

"""# Encoder Model

First we build a simple encoder model, which will be very similar to what you did in MP2. But instead of using a fully connected layer as the output, you should the return the output of your recurrent net (GRU/LSTM) as well as the hidden output. They are used in the decoder later.
"""

## Feel free to change any parameters class definitions as long as you can change the training code, but make sure
## evaluation should get the tensor format it expects
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, max_len):
        super(Encoder, self).__init__()
        # input hyperparameters here:
        global NUM_LAYERS, BIDIRECTIONAL, DROPOUT, HIDDEN_DIM
        self.num_layers = NUM_LAYERS
        self.bidirectional = BIDIRECTIONAL
        self.num_directions = 2 if BIDIRECTIONAL else 1
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.max_len = max_len

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        self.embeds = nn.Embedding(vocab_size, embedding_dim, 0)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = HIDDEN_DIM, num_layers = NUM_LAYERS, dropout = DROPOUT, bidirectional = BIDIRECTIONAL, batch_first=True)
        self.hidden = None
        self.c = None
        
    def forward(self, x, lens):
        '''
        Pseudo-code
        - Pass x through an embedding layer
        - Make sure x is correctly packed before the recurrent net 
        - Pass it through the recurrent net
        - Make sure the output is unpacked correctly
        - return output and hidden states from the recurrent net
        - Feel free to play around with dimensions - the training loop should help you determine the dimensions
        '''
        #x = [batch size, MAX LEN]
        #lens = [batch size]
        embeds = self.embeds(x) # torch.Size([max_len, batch_size, embedding_dim])
        # Embedding dimensions: torch.Size([16, 60, 256])

        embeds = embeds.permute(1, 0, 2) # torch.Size([batch_size, max_len, embedding_dim])

        embeds = nn.utils.rnn.pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False).float()
        embeds.to(device)

        # pass input through recurrent net
        if self.hidden is None:
          lstm_out, (self.hidden, self.c) = self.lstm(embeds)
        else:
          lstm_out, (self.hidden, self.c) = self.lstm(embeds, (self.hidden, self.c))
        lstm_out, out_lens = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.max_len)
        # lstm_out dimensions: torch.Size([60, 16, 256])
        # hidden dimensions: torch.Size([4, 60, 256])

        #hidden = [n_layers*num_directions, batch_size, hidden_dim]
        #c = [n_layers*num_directions, batch_size, hidden_dim]
        #lstm_out = [batch_size, MAX_LEN, hidden_dim*num_directions]
        #out_lens = text_lengths

        # need to provide both last hidden state and output array to decoder      

        # need to translate hidden from 8*batch_size*self.hidden_dim 
        # to 4 * 1 * self.hidden_dim*2 if we're doing bidirectional lstm
        hidden = (self.hidden, self.c)
        if self.bidirectional:
          hidden = tuple([self.format_bidirectional_shape(self.hidden),self.format_bidirectional_shape(self.c)])
        
        '''
        print("Encoder output shape:",lstm_out.shape)
        print("Encoder hidden shape:",self.hidden.shape)
        print("Encoder c shape:",self.c.shape)
        '''

        return lstm_out, hidden

    @staticmethod
    def format_bidirectional_shape(h):
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

"""# Decoder Model
We will implement a Decoder model which uses an attention mechanism. We will implement the decoder as provided in https://arxiv.org/pdf/1409.0473.pdf. **Please read** the links provided above first, at the start of this assignment for review. The pseudo-code for your implementation should be somewhat as follows:



1.   The input is put through an encoder model which gives us the encoder output of shape *(batch_size, max_length, hidden_size)* and the encoder hidden state of shape *(batch_size, hidden_size)*. 
2.   Using the output your encoder you will calculate the score and subsequently the attention using following equations : 
<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_0.jpg" alt="attention equation 0" width="800">
<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">

3. Once you have calculated this attention vector, you pass the original input x through a embedding layer. The output of this embedding layer is concatenated with the attention vector which is passed into a GRU.

4. Finally you pass the output of the GRU into a fully connected layer with an output size same as that vocab, to see the probability of the most possible word.
"""

import torch.nn.functional as F
from torch.autograd import Variable

## Feel free to change any parameters class definitions as long as you can change the training code, but make sure
## evaluation should get the tensor format it expects
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz, tar_len, src_len):
        '''                8029         256           1024      1024      60     '''
        super(Decoder, self).__init__()
        #print("Decoder parameters:", vocab_size, embedding_dim, dec_units, enc_units, batch_sz)
        # import hyperparameters here:
        global NUM_LAYERS, DROPOUT, HIDDEN_DIM, BIDIRECTIONAL
        self.num_layers = NUM_LAYERS
        self.dropout_val = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.bidirectional = BIDIRECTIONAL
        num_directions = 2 if self.bidirectional else 1

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.batch_sz = batch_sz
        self.tar_len = tar_len
        self.src_len = src_len
        
        self.embeds = nn.Embedding(vocab_size, embedding_dim, 0)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = HIDDEN_DIM, num_layers = NUM_LAYERS, dropout = self.dropout_val, bidirectional = BIDIRECTIONAL, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_val)

        
        self.W1 = nn.Linear(self.hidden_dim * num_directions, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim * num_directions, self.hidden_dim, bias=False)
        self.W3 = nn.Linear(self.hidden_dim * num_directions, 1, bias=False)
        
        self.hidden_to_vocab_size = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        
    
    def forward(self, x, decoder_hidden, enc_output):#, src_lens):
        '''
        Pseudo-code
        - Calculate the score using the formula shown above using encoder output and hidden output. 
        Note h_t is the hidden output of the decoder and h_s is the encoder output in the formula
        - Calculate the attention weights using softmax and passing through V - which can be implemented as a fully connected layer
        - Finally find c_t which is a context vector where the shape of context_vector should be (batch_size, hidden_size)
        - You need to unsqueeze the context_vector for concatenating with x as listed in Point 3 above
        - Pass this concatenated tensor to the GRU and follow as specified in Point 4 above

        x          : [60, 1]
        enc_output : [60, 16, 256]
        hidden[0]  : [4, 60, 256]
        hidden[1]  : [4, 60, 256]
        '''

        # last_layer_hidden == (batch_size, 1, hidden_size)
        last_layer_hidden = decoder_hidden[0][-1]
        last_layer_hidden = last_layer_hidden.unsqueeze(1)

        # score shape == (batch_size, max_length, hidden_size)
        score = torch.tanh(self.W1(enc_output) + self.W2(last_layer_hidden))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.W3
        attention_weights = self.softmax(self.W3(score))

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output # torch.Size([batch_size, max_len, hidden_size])
        context_vector = torch.sum(context_vector, 1) # torch.Size([batch_size, hidden_size])

        #x = [batch size, 1]
        #lens = [batch size]
        # embeds shape == (batch_size, 1, hidden_size)
        embeds = self.embeds(x.long())
        embeds.to(device)
        
        # pass embeddings modified with attention vectors through recurrent decoder net
        # decoder_out shape == (batch_size, vocab_size)
        # decoder_hidden shape == (num_layers, batch_size, hidden_size)
        decoder_out, decoder_hidden = self.lstm(embeds, decoder_hidden)
        """
        Decoder output is of shape torch.Size([60, 1, 256])
        Decoder hidden state is of shape torch.Size([4, 60, 256])
        Decoder c state is of shape torch.Size([4, 60, 256])
        """
        decoder_out = decoder_out.squeeze(1)
        decoder_out = self.hidden_to_vocab_size(decoder_out)       
        attention_weights = attention_weights.squeeze(2)

        "        Returns :"
        "        output - shape = (batch_size, vocab)"
        "        hidden state - shape = (num_layers*num_dimensions, batch_size, hidden size)"
        "        attention weights - shape = (batch_size, max_src_len)"

        return decoder_out, decoder_hidden, attention_weights

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    #mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
NUM_LAYERS = 4
BIDIRECTIONAL = False
DROPOUT = 0.1
HIDDEN_DIM = embedding_dim
BIAS = True

## Feel free to change any parameters class definitions as long as you can change the training code, but make sure
## evaluation should get the tensor format it expects, this is only for reference
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, max_length_inp)
decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE, max_length_inp, max_length_tar)

encoder.to(device)
decoder.to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                       lr=0.001)

"""# Train your model

You will train your model here.
*   Pass the source sentence and their corresponding lengths into the encoder
*   Creating the decoder input using <start> tokens
*   Now we find out the decoder outputs conditioned on the previous predicted word usually, but in our training we use teacher forcing. Read more about teacher forcing at https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
"""

EPOCHS = 10

encoder.train()
decoder.train()

for epoch in range(EPOCHS):
    start = time.time()
    
    total_loss = 0
    
    for (batch, (inp, targ, inp_len)) in enumerate(dataset):
        loss = 0
        
        xs, ys, lens = sort_batch(inp, targ, inp_len)
        enc_output, enc_hidden = encoder(xs.to(device), lens)
        dec_hidden = enc_hidden
        #dec_hidden[0].to(device)
        #dec_hidden[1].to(device) # tuple has no attribute '.to(device)'
        
        # use teacher forcing - feeding the target as the next input (via dec_input)
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
        
        # run code below for every timestep in the ys batch
        for t in range(1, ys.size(1)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                         dec_hidden, 
                                         enc_output.to(device))
            
            loss += loss_function(ys[:, t].to(device), predictions.to(device))
            
            dec_input = ys[:, t].unsqueeze(1)
            
        encoder.hidden.detach_()
        encoder.c.detach_()
        encoder.hidden = encoder.hidden.detach()
        encoder.c = encoder.c.detach()
        
        batch_loss = (loss / int(ys.size(1)))
        total_loss += batch_loss

        optimizer.zero_grad()
        
        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.detach().item()))
        
        
    ### TODO: Save checkpoint for model
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

"""# Evaluation


*   We evaluate on the test set.
*   In this evaluation, instead of using the concept of teacher forcing, we use the prediction of the decoder as the input to the decoder for the sequence of outputs.
"""

start = time.time()

encoder.eval()
decoder.eval()

total_loss = 0

final_output = torch.zeros((len(target_tensor_val),max_length_tar))
target_output = torch.zeros((len(target_tensor_val),max_length_tar))

for (batch, (inp, targ, inp_len)) in enumerate(val_dataset):
    loss = 0
    xs, ys, lens = sort_batch(inp, targ, inp_len)
    enc_output, enc_hidden = encoder(xs.to(device), lens)
    dec_hidden = enc_hidden
    dec_hidden[0].to(device)
    dec_hidden[1].to(device)
    
    dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
    curr_output = torch.zeros((ys.size(0), ys.size(1)))
    curr_output[:, 0] = dec_input.squeeze(1)

    for t in range(1, ys.size(1)): # run code below for every timestep in the ys batch
        predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                      dec_hidden, 
                                      enc_output.to(device))
        loss += loss_function(ys[:, t].to(device), predictions.to(device))
        dec_input = torch.argmax(predictions, dim=1).unsqueeze(1)
        curr_output[:, t] = dec_input.squeeze(1)
    final_output[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE] = curr_output
    target_output[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE] = targ
    batch_loss = (loss / int(ys.size(1)))
    total_loss += batch_loss

print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                    total_loss / N_BATCH))
print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

"""# Bleu Score Calculation for evaluation

Read more about Bleu Score at :


1.   https://en.wikipedia.org/wiki/BLEU
2.   https://www.aclweb.org/anthology/P02-1040.pdf

We expect your BLEU Scores to be in the range of for full credit. No partial credit :( 


*   BLEU-1 > 0.14
*   BLEU-2 > 0.08
*   BLEU-3 > 0.02
*   BLEU-4 > 0.15
"""

def get_reference_candidate(target, pred):
  reference = list(target)
  reference = [targ_lang.idx2word[s] for s in np.array(reference[1:])]
  candidate = list(pred)
  candidate = [targ_lang.idx2word[s] for s in np.array(candidate[1:])]
  return reference, candidate

bleu_1 = 0.0
bleu_2 = 0.0
bleu_3 = 0.0
bleu_4 = 0.0
smoother = SmoothingFunction()
save_candidate = []

for i in range(len(target_tensor_val)):
  reference, candidate = get_reference_candidate(target_output[i], final_output[i])
  #print(reference)
  #print(candidate)
  save_candidate.append(candidate)

  bleu_1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
  bleu_2 += sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=smoother.method2)
  bleu_3 += sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=smoother.method3)
  bleu_4 += sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=smoother.method4)

print('Individual 1-gram: %f' % (bleu_1/len(target_tensor_val)))
print('Individual 2-gram: %f' % (bleu_2/len(target_tensor_val)))
print('Individual 3-gram: %f' % (bleu_3/len(target_tensor_val)))
print('Individual 4-gram: %f' % (bleu_4/len(target_tensor_val)))
assert(len(save_candidate)==len(target_tensor_val))

"""# Save File for Submission
You just need to submit your **results.pickle** file to the autograder.
"""

import pickle
from google.colab import drive
drive.mount('/content/drive')

with open('../drive/My Drive/results.pickle', 'wb') as fil:
    pickle.dump(save_candidate, fil)