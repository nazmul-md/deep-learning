#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:25:10 2018

@author: iit
"""
# In[ ]:
"""1"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
random.seed(1024)
# In[ ]:
"""2"""

USE_CUDA = torch.cuda.is_available()
gpus = [0]
#torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
# In[ ]:
"""3"""
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
        
# In[ ]:
"""4"""
def pad_to_batch(batch, w_to_ix): # for bAbI dataset
    fact,q,a = list(zip(*batch))
    max_fact = max([len(f) for f in fact])
    max_len = max([f.size(1) for f in flatten(fact)])
    max_q = max([qq.size(1) for qq in q])
    max_a = max([aa.size(1) for aa in a])
    
    facts, fact_masks, q_p, a_p = [], [], [], []
    for i in range(len(batch)):
        fact_p_t = []
        for j in range(len(fact[i])):
            if fact[i][j].size(1) < max_len:
                fact_p_t.append(torch.cat([fact[i][j], Variable(LongTensor([w_to_ix['<PAD>']] * (max_len - fact[i][j].size(1)))).view(1, -1)], 1))
            else:
                fact_p_t.append(fact[i][j])

        while len(fact_p_t) < max_fact:
            fact_p_t.append(Variable(LongTensor([w_to_ix['<PAD>']] * max_len)).view(1, -1))

        fact_p_t = torch.cat(fact_p_t)
        facts.append(fact_p_t)
        fact_masks.append(torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in fact_p_t]).view(fact_p_t.size(0), -1))

        if q[i].size(1) < max_q:
            q_p.append(torch.cat([q[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_q - q[i].size(1)))).view(1, -1)], 1))
        else:
            q_p.append(q[i])

        if a[i].size(1) < max_a:
            a_p.append(torch.cat([a[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_a - a[i].size(1)))).view(1, -1)], 1))
        else:
            a_p.append(a[i])

    questions = torch.cat(q_p)
    answers = torch.cat(a_p)
    question_masks = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in questions]).view(questions.size(0), -1)
    
    return facts, fact_masks, questions, question_masks, answers
# In[ ]:
"""5"""
def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))
# In[ ]:
"""6"""
def bAbI_data_load(path):
    try:
        data = open(path).readlines()
    except:
        print("Such a file does not exist at %s".format(path))
        return None
    
    data = [d[:-1] for d in data]
    data_p = []
    fact = []
    qa = []
    try:
        for d in data:
            index = d.split(' ')[0]
            if index == '1':
                fact = []
                qa = []
            if '?' in d:
                temp = d.split('\t')
                q = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
                a = temp[1].split() + ['</s>']
                stemp = deepcopy(fact)
                data_p.append([stemp, q, a])
            else:
                tokens = d.replace('.', '').split(' ')[1:] + ['</s>']
                fact.append(tokens)
    except:
        print("Please check the data is right")
        return None
    return data_p
# In[ ]:
"""7"""
projectDir = os.path.dirname(os.path.realpath('__file__'))
trainDir = projectDir + '/dataset/bAbI/en-10k/qa5_three-arg-relations_train.txt';
train_data = bAbI_data_load(trainDir)
# In[ ]:
"""8"""
train_data[0]
# In[ ]:
"""9"""
fact,q,a = list(zip(*train_data))
# In[ ]:
"""10"""
vocab = list(set(flatten(flatten(fact)) + flatten(q) + flatten(a)))
# In[ ]:
"""11"""
word2index={'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}
# In[ ]:
"""12"""
len(word2index)

# In[ ]:
"""13"""
for t in train_data:
    for i,fact in enumerate(t[0]):
        t[0][i] = prepare_sequence(fact, word2index).view(1, -1)
    
    t[1] = prepare_sequence(t[1], word2index).view(1, -1)
    t[2] = prepare_sequence(t[2], word2index).view(1, -1)
# In[ ]:
"""14"""
class DMN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(DMN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0) #sparse=True)
        self.input_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.question_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        self.gate = nn.Sequential(
                            nn.Linear(hidden_size * 4, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1),
                            nn.Sigmoid()
                        )
        
        self.attention_grucell =  nn.GRUCell(hidden_size, hidden_size)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.answer_grucell = nn.GRUCell(hidden_size * 2, hidden_size)
        self.answer_fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_p)
        
    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(1, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden
    
    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])
        
        for name, param in self.input_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.question_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.gate.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.attention_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.memory_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.answer_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        
        nn.init.xavier_normal(self.answer_fc.state_dict()['weight'])
        self.answer_fc.bias.data.fill_(0)
        
    def forward(self, facts, fact_masks, questions, question_masks, num_decode, episodes=3, is_training=False):
        """
        facts : (B,T_C,T_I) / LongTensor in List # batch_size, num_of_facts, length_of_each_fact(padded)
        fact_masks : (B,T_C,T_I) / ByteTensor in List # batch_size, num_of_facts, length_of_each_fact(padded)
        questions : (B,T_Q) / LongTensor # batch_size, question_length
        question_masks : (B,T_Q) / ByteTensor # batch_size, question_length
        """
        # Input Module
        C = [] # encoded facts
        for fact, fact_mask in zip(facts, fact_masks):
            embeds = self.embed(fact)
            if is_training:
                embeds = self.dropout(embeds)
            hidden = self.init_hidden(fact)
            outputs, hidden = self.input_gru(embeds, hidden)
            real_hidden = []

            for i, o in enumerate(outputs): # B,T,D
                real_length = fact_mask[i].data.tolist().count(0) 
                real_hidden.append(o[real_length - 1])

            C.append(torch.cat(real_hidden).view(fact.size(0), -1).unsqueeze(0))
        
        encoded_facts = torch.cat(C) # B,T_C,D
        
        # Question Module
        embeds = self.embed(questions)
        if is_training:
            embeds = self.dropout(embeds)
        hidden = self.init_hidden(questions)
        outputs, hidden = self.question_gru(embeds, hidden)
        
        if isinstance(question_masks, torch.autograd.Variable):
            real_question = []
            for i, o in enumerate(outputs): # B,T,D
                real_length = question_masks[i].data.tolist().count(0) 
                real_question.append(o[real_length - 1])
            encoded_question = torch.cat(real_question).view(questions.size(0), -1) # B,D
        else: # for inference mode
            encoded_question = hidden.squeeze(0) # B,D
            
        # Episodic Memory Module
        memory = encoded_question
        T_C = encoded_facts.size(1)
        B = encoded_facts.size(0)
        for i in range(episodes):
            hidden = self.init_hidden(encoded_facts.transpose(0, 1)[0]).squeeze(0) # B,D
            for t in range(T_C):
                #TODO: fact masking
                #TODO: gate function => softmax
                z = torch.cat([
                                    encoded_facts.transpose(0, 1)[t] * encoded_question, # B,D , element-wise product
                                    encoded_facts.transpose(0, 1)[t] * memory, # B,D , element-wise product
                                    torch.abs(encoded_facts.transpose(0,1)[t] - encoded_question), # B,D
                                    torch.abs(encoded_facts.transpose(0,1)[t] - memory) # B,D
                                ], 1)
                g_t = self.gate(z) # B,1 scalar
                hidden = g_t * self.attention_grucell(encoded_facts.transpose(0, 1)[t], hidden) + (1 - g_t) * hidden
                
            e = hidden
            memory = self.memory_grucell(e, memory)
        
        # Answer Module
        answer_hidden = memory
        start_decode = Variable(LongTensor([[word2index['<s>']] * memory.size(0)])).transpose(0, 1)
        y_t_1 = self.embed(start_decode).squeeze(1) # B,D
        
        decodes = []
        for t in range(num_decode):
            answer_hidden = self.answer_grucell(torch.cat([y_t_1, encoded_question], 1), answer_hidden)
            decodes.append(F.log_softmax(self.answer_fc(answer_hidden),1))
        return torch.cat(decodes, 1).view(B * num_decode, -1)
    
# In[ ]:
"""15"""
HIDDEN_SIZE = 80
BATCH_SIZE = 64
LR = 0.001
EPOCH = 50
NUM_EPISODE = 3
EARLY_STOPPING = False    
# In[ ]:
"""16"""
model = DMN(len(word2index), HIDDEN_SIZE, len(word2index))
model.init_weight()
if USE_CUDA:
    model = model.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR)
# In[ ]:
"""17"""
for epoch in range(EPOCH):
    losses = []
    if EARLY_STOPPING: 
        break
        
    for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        facts, fact_masks, questions, question_masks, answers = pad_to_batch(batch, word2index)
        
        model.zero_grad()
        pred = model(facts, fact_masks, questions, question_masks, answers.size(1), NUM_EPISODE, True)
        loss = loss_function(pred, answers.view(-1))
       
        losses.append(loss.data.tolist()[0])
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
            
            if np.mean(losses) < 0.01:
                EARLY_STOPPING = True
                print("Early Stopping!")
                break
            losses = []
            
# In[ ]:
"""18"""
def pad_to_fact(fact, x_to_ix): # this is for inference
    
    max_x = max([s.size(1) for s in fact])
    x_p = []
    for i in range(len(fact)):
        if fact[i].size(1) < max_x:
            x_p.append(torch.cat([fact[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - fact[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(fact[i])
        
    fact = torch.cat(x_p)
    fact_mask = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in fact]).view(fact.size(0), -1)
    return fact, fact_mask            
# In[ ]:
"""19"""
projectDir = os.path.dirname(os.path.realpath('__file__'))
testDir = projectDir + '/dataset/bAbI/en-10k/qa5_three-arg-relations_test.txt';
test_data = bAbI_data_load(testDir)
# In[ ]:
"""20"""
for t in test_data:
    for i, fact in enumerate(t[0]):
        t[0][i] = prepare_sequence(fact, word2index).view(1, -1)
    
    t[1] = prepare_sequence(t[1], word2index).view(1, -1)
    t[2] = prepare_sequence(t[2], word2index).view(1, -1)
# In[ ]:
"""21"""    
accuracy = 0
for t in test_data:
    fact, fact_mask = pad_to_fact(t[0], word2index)
    question = t[1]
    question_mask = Variable(ByteTensor([0] * t[1].size(1)), volatile=False).unsqueeze(0)
    answer = t[2].squeeze(0)
    
    model.zero_grad()
    pred = model([fact], [fact_mask], question, question_mask, answer.size(0), NUM_EPISODE)
    if pred.max(1)[1].data.tolist() == answer.data.tolist():
        accuracy += 1

print(accuracy/len(test_data) * 100)    
# In[ ]:
"""22"""  
t = random.choice(test_data)
fact, fact_mask = pad_to_fact(t[0], word2index)
question = t[1]
question_mask = Variable(ByteTensor([0] * t[1].size(1)), volatile=False).unsqueeze(0)
answer = t[2].squeeze(0)

model.zero_grad()
pred = model([fact], [fact_mask], question, question_mask, answer.size(0), NUM_EPISODE)

print("Facts : ")
print('\n'.join([' '.join(list(map(lambda x: index2word[x],f))) for f in fact.data.tolist()]))
print("")
print("Question : ",' '.join(list(map(lambda x: index2word[x], question.data.tolist()[0]))))
print("")
print("Answer : ",' '.join(list(map(lambda x: index2word[x], answer.data.tolist()))))
print("Prediction : ",' '.join(list(map(lambda x: index2word[x], pred.max(1)[1].data.tolist()))))