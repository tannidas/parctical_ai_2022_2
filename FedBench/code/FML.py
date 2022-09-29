#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import random
from torch.autograd import Variable
import copy
from torch import nn, optim
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
import time
import math
import re
import json
import sys

sys.path.append('./utils/')
from model import *
from utils import *


# In[ ]:


## If you want to experiment with a different seed value, change 'trial_times'
trial_times = 1
SEED = 42 + trial_times -1
fix_seed(SEED)


# In[ ]:


class Argments():
  def __init__(self):
    self.batch_size = 20 
    self.test_batch = 1000
    self.global_epochs = 300
    self.local_epochs = 2
    self.lr = 10**(-3)
    self.momentum = 0.9
    self.weight_decay = 10**-4.0
    self.clip = 20.0
    self.partience = 300
    self.worker_num = 20
    self.participation_rate = 1
    self.sample_num = int(self.worker_num * self.participation_rate)
    self.total_data_rate = 1
    self.unlabeleddata_size = 1000
    self.device = device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')
    self.criterion_ce = nn.CrossEntropyLoss()
    self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    ## If you use MNIST or CIFAR-10, the degree of data heterogeneity can be changed by changing alpha_label and alpha_size.
    self.alpha_label = 0.5
    self.alpha_size = 10
    
    ## Select a dataset from 'FEMNIST','Shakespeare','Sent140','MNIST', or 'CIFAR-10'.
    self.dataset_name = 'FEMNIST'


args = Argments()


# In[ ]:


if args.dataset_name=='FEMNIST':
    from femnist_dataset import *
    args.num_classes = 62
    model_name = "CNN(num_classes=args.num_classes)"
    
elif args.dataset_name=='Shakespeare':
    from shakespeare_dataset import *
    model_name = "RNN()"
    
elif args.dataset_name=='Sent140':
    from sent140_dataset import *
    from utils_sent140 import *
    VOCAB_DIR = '../models/embs.json'
    _, indd, vocab = get_word_emb_arr(VOCAB_DIR)
    model_name = "RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False)"
    
elif args.dataset_name=='MNIST':
    from mnist_dataset import *
    args.num_classes = 10
    model_name = "CNN(num_classes=args.num_classes)"
    
elif args.dataset_name=='CIFAR-10':
    from cifar10_dataset import *
    model_name = "vgg13()"
    
else:
    print('Error: The name of the dataset is incorrect. Please re-set the "dataset_name".')


# In[ ]:


federated_trainset,federated_valset,federated_testset,unlabeled_dataset = get_dataset(args, unlabeled_data=True)


# In[ ]:


class Server():
  def __init__(self):
    self.global_model = eval(model_name)

  def create_worker(self,federated_trainset,federated_valset,federated_testset):
    workers = []
    for i in range(args.worker_num):
      workers.append(Worker(federated_trainset[i],federated_valset[i],federated_testset[i]))
    return workers

  def sample_worker(self,workers):
    sample_worker = []
    sample_worker_num = random.sample(range(args.worker_num),args.sample_num)
    for i in sample_worker_num:
      sample_worker.append(workers[i])
    return sample_worker


  def send_model(self,workers):
    nums = 0
    for worker in workers:
      nums += worker.train_data_num

    for worker in workers:
      worker.aggregation_weight = 1.0*worker.train_data_num/nums
      worker.global_model = copy.deepcopy(self.global_model)

  def aggregate_model(self,workers):   
    new_params = OrderedDict()
    for i,worker in enumerate(workers):
      worker_state = worker.global_model.state_dict()
      for key in worker_state.keys():
        if i==0:
          new_params[key] = worker_state[key]*worker.aggregation_weight
        else:
          new_params[key] += worker_state[key]*worker.aggregation_weight
      worker.global_model = worker.global_model.to('cpu')
      del worker.global_model
    self.global_model.load_state_dict(new_params)


# In[ ]:


class Worker():
  def __init__(self,trainset,valset,testset):
    self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2)
    self.valloader = torch.utils.data.DataLoader(valset,batch_size=args.test_batch,shuffle=False,num_workers=2)
    self.testloader = torch.utils.data.DataLoader(testset,batch_size=args.test_batch,shuffle=False,num_workers=2)
    self.local_model = eval(model_name)
    self.global_model = None
    self.train_data_num = len(trainset)
    self.test_data_num = len(testset)
    self.aggregation_weight = None

  def local_train(self):
    if args.dataset_name=='Sent140':
        self.local_model.train()
        self.global_model.train()
        self.local_model = self.local_model.to(args.device)
        self.global_model = self.global_model.to(args.device)
        local_hidden_train = self.local_model.init_hidden(args.batch_size)
        global_hidden_train = self.global_model.init_hidden(args.batch_size)
        local_optimizer = optim.SGD(self.local_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        global_optimizer = optim.SGD(self.global_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        for epoch in range(args.local_epochs):
          running_loss = 0.0
          correct = 0
          count = 0
          for (data,labels) in self.trainloader:
            data, labels = process_x(data, indd), process_y(labels, indd)
            if args.batch_size != 1 and data.shape[0] != args.batch_size:
              break
            data,labels = torch.from_numpy(data).to(args.device), torch.from_numpy(labels).to(args.device)    
            local_optimizer.zero_grad()
            global_optimizer.zero_grad()
            local_hidden_train = repackage_hidden(local_hidden_train)
            global_hidden_train = repackage_hidden(global_hidden_train)
            local_outputs, local_hidden_train = self.local_model(data, local_hidden_train) 
            global_outputs, global_hidden_train = self.global_model(data, global_hidden_train) 

            #train local_model
            ce_loss = args.criterion_ce(local_outputs.t(), torch.max(labels, 1)[1])
            kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim = 1),F.softmax(Variable(global_outputs), dim=1))
            loss = ce_loss + kl_loss
            running_loss += loss.item()
            _, predicted = torch.max(local_outputs.t(), 1)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
            count += len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)
            local_optimizer.step()

            #train global_model
            ce_loss = args.criterion_ce(global_outputs.t(), torch.max(labels, 1)[1])
            kl_loss = args.criterion_kl(F.log_softmax(global_outputs, dim = 1),F.softmax(Variable(local_outputs), dim=1))
            loss = ce_loss + kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), args.clip)
            global_optimizer.step()

        return 100.0*correct/count,running_loss/len(self.trainloader)
    
    else:
        self.local_model = self.local_model.to(args.device)
        self.global_model = self.global_model.to(args.device)
        local_optimizer = optim.SGD(self.local_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        global_optimizer = optim.SGD(self.global_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        self.local_model.train()
        self.global_model.train()
        for epoch in range(args.local_epochs):
          running_loss = 0.0
          correct = 0
          count = 0
          for (data,labels) in self.trainloader:
            data,labels = Variable(data),Variable(labels)
            data,labels = data.to(args.device),labels.to(args.device)
            local_optimizer.zero_grad()
            global_optimizer.zero_grad()
            local_outputs = self.local_model(data)
            global_outputs = self.global_model(data)
            #train local_model
            ce_loss = args.criterion_ce(local_outputs,labels)
            kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim = 1),F.softmax(Variable(global_outputs), dim=1))
            loss = ce_loss + kl_loss
            running_loss += loss.item()
            predicted = torch.argmax(local_outputs,dim=1)
            correct += (predicted==labels).sum().item()
            count += len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)
            local_optimizer.step()

            #train global_model
            ce_loss = args.criterion_ce(global_outputs,labels)
            kl_loss = args.criterion_kl(F.log_softmax(global_outputs, dim = 1),F.softmax(Variable(local_outputs), dim=1))
            loss = ce_loss + kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), args.clip)
            global_optimizer.step()

        return 100.0*correct/count,running_loss/len(self.trainloader)        


  def validate(self):
    acc,loss = test(self.local_model,args.criterion_ce,self.valloader)
    self.local_model = self.local_model.to('cpu')
    return acc,loss


# In[ ]:


def train(model,criterion,trainloader,epochs):
  if args.dataset_name=='Sent140':
      model.train()
      hidden_train = model.init_hidden(args.batch_size)
      optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
      for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        count = 0
        for (data,labels) in trainloader:
          data, labels = process_x(data, indd), process_y(labels, indd)
          if args.batch_size != 1 and data.shape[0] != args.batch_size:
            break
          data,labels = torch.from_numpy(data).to(args.device), torch.from_numpy(labels).to(args.device)
          optimizer.zero_grad()
          hidden_train = repackage_hidden(hidden_train)
          outputs, hidden_train = model(data, hidden_train) 
          loss = criterion(outputs.t(), torch.max(labels, 1)[1])
          running_loss += loss.item()
          _, predicted = torch.max(outputs.t(), 1)
          correct += (predicted == torch.max(labels, 1)[1]).sum().item()
          count += len(labels)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
          optimizer.step()

      return 100.0*correct/count,running_loss/len(trainloader)



  else:
      optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
      model.train()
      for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        count = 0
        for (data,labels) in trainloader:
          data,labels = Variable(data),Variable(labels)
          data,labels = data.to(args.device),labels.to(args.device)
          optimizer.zero_grad()
          outputs = model(data)
          loss = criterion(outputs,labels)
          running_loss += loss.item()
          predicted = torch.argmax(outputs,dim=1)
          correct += (predicted==labels).sum().item()
          count += len(labels)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
          optimizer.step()

      return 100.0*correct/count,running_loss/len(trainloader)       


# In[ ]:


def test(model,criterion,testloader):
  if args.dataset_name=='Sent140':
      model.eval()
      hidden_test = model.init_hidden(args.test_batch)
      running_loss = 0.0
      correct = 0
      count = 0
      for (data,labels) in testloader:
        data, labels = process_x(data, indd), process_y(labels, indd)
        if args.test_batch != 1 and data.shape[0] != args.test_batch:
          break
        data,labels = torch.from_numpy(data).to(args.device), torch.from_numpy(labels).to(args.device)
        hidden_test = repackage_hidden(hidden_test)
        outputs, hidden_test = model(data, hidden_test) 
        running_loss += criterion(outputs.t(), torch.max(labels, 1)[1]).item()
        _, predicted = torch.max(outputs.t(), 1)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()
        count += len(labels)

      accuracy = 100.0*correct/count
      loss = running_loss/len(testloader)


      return accuracy,loss


  else:
      model.eval()
      running_loss = 0.0
      correct = 0
      count = 0
      for (data,labels) in testloader:
        data,labels = data.to(args.device),labels.to(args.device)
        outputs = model(data)
        running_loss += criterion(outputs,labels).item()
        predicted = torch.argmax(outputs,dim=1)
        correct += (predicted==labels).sum().item()
        count += len(labels)

      accuracy = 100.0*correct/count
      loss = running_loss/len(testloader)


      return accuracy,loss


# In[ ]:


server = Server()
workers = server.create_worker(federated_trainset,federated_valset,federated_testset)
acc_train = []
loss_train = []
acc_valid = []
loss_valid = []

early_stopping = Early_Stopping(args.partience)

start = time.time()

for epoch in range(args.global_epochs):
  sample_worker = server.sample_worker(workers)
  server.send_model(sample_worker)

  acc_train_avg = 0.0
  loss_train_avg = 0.0
  acc_valid_avg = 0.0
  loss_valid_avg = 0.0
  for worker in sample_worker:
    acc_train_tmp,loss_train_tmp = worker.local_train()
    acc_valid_tmp,loss_valid_tmp = worker.validate()
    acc_train_avg += acc_train_tmp/len(sample_worker)
    loss_train_avg += loss_train_tmp/len(sample_worker)
    acc_valid_avg += acc_valid_tmp/len(sample_worker)
    loss_valid_avg += loss_valid_tmp/len(sample_worker)
  server.aggregate_model(sample_worker)
  '''
  server.model.to(args.device)
  for worker in workers:
    acc_valid_tmp,loss_valid_tmp = test(server.model,args.criterion,worker.valloader)
    acc_valid_avg += acc_valid_tmp/len(workers)
    loss_valid_avg += loss_valid_tmp/len(workers)
  server.model.to('cpu')
  '''
  print('Epoch{}  loss:{}  accuracy:{}'.format(epoch+1,loss_valid_avg,acc_valid_avg))
  acc_train.append(acc_train_avg)
  loss_train.append(loss_train_avg)
  acc_valid.append(acc_valid_avg)
  loss_valid.append(loss_valid_avg)

  if early_stopping.validate(loss_valid_avg):
    print('Early Stop')
    break
    
end = time.time()


# In[ ]:


print('train time：{}[s]'.format(end-start))


# In[ ]:


acc_test = []
loss_test = []

start = time.time()

for i,worker in enumerate(workers):
  worker.global_model = copy.deepcopy(server.global_model)
  _,_ = worker.local_train()
  acc_tmp,loss_tmp = test(worker.local_model,args.criterion_ce,worker.testloader)
  acc_test.append(acc_tmp)
  loss_test.append(loss_tmp)
  print('Worker{} accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))
  worker.local_model = worker.local_model.to('cpu')
  worker.global_model = worker.global_model.to('cpu')
  del worker.global_model

end = time.time()

acc_test_avg = sum(acc_test)/len(acc_test)
loss_test_avg = sum(loss_test)/len(loss_test)
print('Test  loss:{}  accuracy:{}'.format(loss_test_avg,acc_test_avg))


# In[ ]:


filename = 'FML_{}'.format(args.dataset_name)
result_path = '../result/'


# In[ ]:


acc_train = pd.DataFrame(acc_train)
loss_train = pd.DataFrame(loss_train)
acc_valid = pd.DataFrame(acc_valid)
loss_valid = pd.DataFrame(loss_valid)

acc_test = pd.DataFrame(acc_test)
loss_test = pd.DataFrame(loss_test)


acc_train.to_csv(result_path+filename+'_train_acc.csv',index=False, header=False)
loss_train.to_csv(result_path+filename+'_train_loss.csv',index=False, header=False)
acc_valid.to_csv(result_path+filename+'_valid_acc.csv',index=False, header=False)
loss_valid.to_csv(result_path+filename+'_valid_loss.csv',index=False, header=False)
acc_test.to_csv(result_path+filename+'_test_acc.csv',index=False, header=False)
loss_test.to_csv(result_path+filename+'_test_loss.csv',index=False, header=False)


# In[ ]:




