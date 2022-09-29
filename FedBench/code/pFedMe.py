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
    self.lamda = 15
    self.K = 5
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
    self.criterion = nn.CrossEntropyLoss()
    
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


class pFedMeOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                localweight.data = localweight.data.to(args.device)
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
                localweight.data = localweight.data.to('cpu')
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']


# In[ ]:


class Server():
  def __init__(self):
    self.model = eval(model_name)

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
      worker.model = copy.deepcopy(self.model)
      worker.personalized_model = copy.deepcopy(self.model)
      worker.local_model = copy.deepcopy(self.model)

  def aggregate_model(self,workers):   
    new_params = OrderedDict()
    for i,worker in enumerate(workers):
      worker_state = worker.model.state_dict()
      for key in worker_state.keys():
        if i==0:
          new_params[key] = worker_state[key]*worker.aggregation_weight
        else:
          new_params[key] += worker_state[key]*worker.aggregation_weight
    self.model.load_state_dict(new_params)
    
  def send_parameters(self,workers):
    nums = 0
    for worker in workers:
      nums += worker.train_data_num
    for worker in workers:
        worker.aggregation_weight = 1.0*worker.train_data_num/nums
        worker.set_parameters(self.model)


# In[ ]:


class Worker():
  def __init__(self,trainset,valset,testset):
    self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2)
    self.valloader = torch.utils.data.DataLoader(valset,batch_size=args.test_batch,shuffle=False,num_workers=2)
    self.testloader = torch.utils.data.DataLoader(testset,batch_size=args.test_batch,shuffle=False,num_workers=2)
    #self.iter_trainloader = iter(self.trainloader)
    self.model = eval(model_name)
    self.local_model = copy.deepcopy(list(self.model.parameters()))
    self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
    self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
    self.train_data_num = len(trainset)
    self.test_data_num = len(testset)
    
    self.optimizer = pFedMeOptimizer(self.model.parameters(),lr=args.lr,lamda=args.lamda)
    
  def set_parameters(self, model):
    for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
        old_param.data = new_param.data.clone()
        local_param.data = new_param.data.clone()
    #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])
    
  def get_next_train_batch(self):
    try:
        # Samples a new batch for persionalizing
        (X, y) = next(self.iter_trainloader)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        self.iter_trainloader = iter(self.trainloader)
        (X, y) = next(self.iter_trainloader)
    return (X.to(args.device), y.to(args.device))    
  '''
  def local_train(self):
    self.model.train()
    optimizer = pFedMeOptimizer(self.model.parameters(),lr=args.lr,lamda=args.lamda)
    for epoch in range(args.local_epochs):
        running_loss = 0.0
        correct = 0
        count = 0
        data,labels = self.get_next_train_batch()
        for i in range(args.K):
            self.model = self.model.to(args.device)
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = args.criterion(outputs,labels)
            running_loss += loss.item()
            predicted = torch.argmax(outputs,dim=1)
            correct += (predicted==labels).sum().item()
            count += len(labels)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
            self.model = self.model.to('cpu')
            personal_model,_ = optimizer.step(self.local_model)
        
        for new_param,localweight in zip(personal_model,self.local_model.parameters()):
            localweight.data = localweight.data - args.lamda * args.lr * (localweight.data - new_param.data)    
    
    del self.model
    
    update_parameters(self.personalized_model,personal_model)
    return 100.0*correct/count,running_loss/len(self.trainloader)
  '''

  def local_train(self):
    if args.dataset_name=='Sent140':
        self.model.train()
        self.model = self.model.to(args.device)
        hidden_train = self.model.init_hidden(args.batch_size)
        for epoch in range(args.local_epochs):
            running_loss = 0.0
            correct = 0
            count = 0
            for (data,labels) in self.trainloader:
                self.model.train()
                data, labels = process_x(data, indd), process_y(labels, indd)
                if args.batch_size != 1 and data.shape[0] != args.batch_size:
                    break
                data,labels = torch.from_numpy(data).to(args.device), torch.from_numpy(labels).to(args.device)
                for k in range(args.K):
                    self.optimizer.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    outputs, hidden_train = self.model(data, hidden_train)
                    loss = args.criterion(outputs.t(), torch.max(labels, 1)[1])
                    if k==(args.K-1):
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.t(), 1)
                        correct += (predicted == torch.max(labels, 1)[1]).sum().item()
                        count += len(labels)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    self.persionalized_model_bar,_ = self.optimizer.step(self.local_model)

                for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                    localweight.data = localweight.data.to(args.device)
                    localweight.data = localweight.data - args.lamda* args.lr * (localweight.data - new_param.data)
                    localweight.data = localweight.data.to('cpu')

        self.update_parameters(self.local_model)

        self.model = self.model.to('cpu')
        for personal_weight, localweight in zip(self.persionalized_model_bar, self.local_model):
            personal_weight.data = personal_weight.data.to('cpu')
            localweight.data = localweight.data.to('cpu')

        return 100.0*correct/count,running_loss/len(self.trainloader)
    
    else:
        self.model.train() 
        self.model = self.model.to(args.device)       
        for epoch in range(args.local_epochs):
            running_loss = 0.0
            correct = 0
            count = 0
            for (data,labels) in self.trainloader:
                self.model.train()
                data,labels = Variable(data),Variable(labels)
                data,labels = data.to(args.device),labels.to(args.device)
                for k in range(args.K):
                    self.optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = args.criterion(outputs,labels)
                    if k==(args.K-1):
                        running_loss += loss.item()
                        predicted = torch.argmax(outputs,dim=1)
                        correct += (predicted==labels).sum().item()
                        count += len(labels)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    self.persionalized_model_bar,_ = self.optimizer.step(self.local_model)

                for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                    localweight.data = localweight.data.to(args.device)
                    localweight.data = localweight.data - args.lamda* args.lr * (localweight.data - new_param.data)
                    localweight.data = localweight.data.to('cpu')

        self.update_parameters(self.local_model)

        self.model = self.model.to('cpu')
        for personal_weight, localweight in zip(self.persionalized_model_bar, self.local_model):
            personal_weight.data = personal_weight.data.to('cpu')
            localweight.data = localweight.data.to('cpu')

        return 100.0*correct/count,running_loss/len(self.trainloader)


  def validate(self):
    self.model.eval()
    self.update_parameters(self.persionalized_model_bar)
    self.model = self.model.to(args.device)
    acc,loss = test(self.model,args.criterion,self.valloader)
    self.model = self.model.to('cpu')
    self.update_parameters(self.local_model)
    return acc,loss


  def test(self):
    self.model.eval()
    self.update_parameters(self.persionalized_model_bar)
    self.model = self.model.to(args.device)
    acc,loss = test(self.model,args.criterion,self.testloader)
    self.model = self.model.to('cpu')
    self.update_parameters(self.local_model)
    return acc,loss


  def update_parameters(self, new_params):
    for param , new_param in zip(self.model.parameters(), new_params):
      param.data = new_param.data.clone()


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
  server.send_parameters(sample_worker)

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


acc_test_personalized = []
loss_test_personalized = []

start = time.time()

for i,worker in enumerate(workers):
  acc_tmp,loss_tmp = worker.test()
  acc_test_personalized.append(acc_tmp)
  loss_test_personalized.append(loss_tmp)
  print('Worker{} accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))

end = time.time()

acc_test_personalized_avg = sum(acc_test_personalized)/len(acc_test_personalized)
loss_test_personalized_avg = sum(loss_test_personalized)/len(loss_test_personalized)
print('Test(personalized)  loss:{}  accuracy:{}'.format(loss_test_personalized_avg,acc_test_personalized_avg))


# In[ ]:


acc_test_global = []
loss_test_global = []

start = time.time()

for i,worker in enumerate(workers):
  server.model = server.model.to(args.device)
  acc_tmp,loss_tmp = test(server.model,args.criterion,worker.testloader)
  acc_test_global.append(acc_tmp)
  loss_test_global.append(loss_tmp)
  print('Worker{} accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))

server.model = server.model.to('cpu')

end = time.time()

acc_test_global_avg = sum(acc_test_global)/len(acc_test_global)
loss_test_global_avg = sum(loss_test_global)/len(loss_test_global)
print('Test(global)  loss:{}  accuracy:{}'.format(loss_test_global_avg,acc_test_global_avg))


# In[ ]:


args.local_epochs = 2


# In[ ]:


acc_tune_test_global = []
loss_tune_test_global = []

start = time.time()

for i,worker in enumerate(workers):
  worker.model = copy.deepcopy(server.model)
  worker.model = worker.model.to(args.device)
  _,_ = train(worker.model,args.criterion,worker.trainloader,args.local_epochs)
  acc_tmp,loss_tmp = test(worker.model,args.criterion,worker.testloader)
  acc_tune_test_global.append(acc_tmp)
  loss_tune_test_global.append(loss_tmp)
  print('Worker{} accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))
  worker.model = worker.model.to('cpu')
  del worker.model

end = time.time()

acc_tune_test_global_avg = sum(acc_tune_test_global)/len(acc_tune_test_global)
loss_tune_test_global_avg = sum(loss_tune_test_global)/len(loss_tune_test_global)
print('Test_fine-tune(global)  loss:{}  accuracy:{}'.format(loss_tune_test_global_avg,acc_tune_test_global_avg))


# In[ ]:


filename = 'pFedMe_{}'.format(args.dataset_name)
result_path = '../result/'


# In[ ]:


acc_train = pd.DataFrame(acc_train)
loss_train = pd.DataFrame(loss_train)
acc_valid = pd.DataFrame(acc_valid)
loss_valid = pd.DataFrame(loss_valid)

acc_test_global = pd.DataFrame(acc_test_global)
loss_test_global = pd.DataFrame(loss_test_global)
acc_test_personalized = pd.DataFrame(acc_test_personalized)
loss_test_personalized = pd.DataFrame(loss_test_personalized)
acc_tune_test_global = pd.DataFrame(acc_tune_test_global)
loss_tune_test_global = pd.DataFrame(loss_tune_test_global)


acc_train.to_csv(result_path+filename+'_train_acc.csv',index=False, header=False)
loss_train.to_csv(result_path+filename+'_train_loss.csv',index=False, header=False)
acc_valid.to_csv(result_path+filename+'_valid_acc.csv',index=False, header=False)
loss_valid.to_csv(result_path+filename+'_valid_loss.csv',index=False, header=False)
acc_test_global.to_csv(result_path+filename+'_test_global_acc.csv',index=False, header=False)
loss_test_global.to_csv(result_path+filename+'_test_global_loss.csv',index=False, header=False)
acc_test_personalized.to_csv(result_path+filename+'_test_personalized_acc.csv',index=False, header=False)
loss_test_personalized.to_csv(result_path+filename+'_test_personalized_loss.csv',index=False, header=False)
acc_tune_test_global.to_csv(result_path+filename+'_tune_test_global_acc.csv',index=False, header=False)
loss_tune_test_global.to_csv(result_path+filename+'_tune_test_global_loss.csv',index=False, header=False)

