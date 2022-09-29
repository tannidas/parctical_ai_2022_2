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
    self.cluster_list = [1,2,3,4]
    self.cluster_num = None
    self.turn_of_cluster_num = [0,150,225,275]
    self.turn_of_replacement_model = list(range(self.global_epochs))
    self.unlabeleddata_size = 1000
    self.device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')
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
  def __init__(self,unlabeled_dataset):
    self.cluster = None
    self.models = None
    self.unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset,batch_size=args.batch_size,shuffle=False,num_workers=2)

  def create_worker(self,federated_trainset,federated_valset,federated_testset):
    workers = []
    for i in range(args.worker_num):
      workers.append(Worker(i,federated_trainset[i],federated_valset[i],federated_testset[i],))
    return workers

  def sample_worker(self,workers):
    sample_worker = []
    sample_worker_num = random.sample(range(args.worker_num),args.sample_num)
    for i in sample_worker_num:
      sample_worker.append(workers[i])
    return sample_worker

  def collect_model(self,workers):
    self.models = [None]*args.worker_num
    for worker in workers:
      self.models[worker.id] = copy.deepcopy(worker.local_model)

  def send_model(self,workers):
    for worker in workers:
      worker.local_model = copy.deepcopy(self.models[worker.id])
      worker.other_model = copy.deepcopy(self.models[worker.other_model_id])
        
  def return_model(self,workers):
    for worker in workers:
      worker.local_model = copy.deepcopy(self.models[worker.local_model_id])
      worker.local_model_id = worker.id
    del self.models
    
  def aggregate_model(self,workers):   
    new_params = []
    train_model_id = []
    train_model_id_count = []
    for worker in workers:
      worker_state = worker.local_model.state_dict()
      if worker.id in train_model_id:
        i = train_model_id.index(worker.id)
        for key in worker_state.keys():
          new_params[i][key] += worker_state[key]
        train_model_id_count[i] += 1
      else:
        new_params.append(OrderedDict())
        train_model_id.append(worker.id)
        train_model_id_count.append(1)
        i = train_model_id.index(worker.id)
        for key in worker_state.keys():
          new_params[i][key] = worker_state[key]
        
      worker_state = worker.other_model.state_dict()
      if worker.other_model_id in train_model_id:
        i = train_model_id.index(worker.other_model_id)
        for key in worker_state.keys():
          new_params[i][key] += worker_state[key]
        train_model_id_count[i] += 1
      else:
        new_params.append(OrderedDict())
        train_model_id.append(worker.other_model_id)
        train_model_id_count.append(1)
        i = train_model_id.index(worker.other_model_id)
        for key in worker_state.keys():
          new_params[i][key] = worker_state[key]
        
      worker.local_model = worker.local_model.to('cpu')
      worker.other_model = worker.other_model.to('cpu')
      del worker.local_model,worker.other_model
    
    for i,model_id in enumerate(train_model_id):
      for key in new_params[i].keys():
        new_params[i][key] = new_params[i][key]/train_model_id_count[i]
      self.models[model_id].load_state_dict(new_params[i])
      
  '''clustering by kmeans'''  
  def clustering(self,workers):
    if args.cluster_num==1:
        pred = [0]*len(workers)
        worker_id_list = []
        for worker in workers:
            worker_id_list.append(worker.id)
    else:
        if args.dataset_name=='Sent140':
            with torch.no_grad():
                worker_softmax_targets = [[] for i in range(len(workers))]
                worker_id_list = []
                count = 0
                for i,model in enumerate(self.models):
                  if model==None:
                    pass
                  else:
                    model = model.to(args.device)
                    model.eval()
                    hidden_test = model.init_hidden(args.batch_size)
                    for data,_ in self.unlabeled_dataloader:
                      data = process_x(data, indd)
                      if args.batch_size != 1 and data.shape[0] != args.batch_size:
                        break
                      data = torch.from_numpy(data).to(args.device)
                      hidden_test = repackage_hidden(hidden_test)
                      outputs, hidden_test = model(data, hidden_test)
                      worker_softmax_targets[count].append(outputs.to('cpu').detach().numpy())
                    worker_softmax_targets[count] = np.array(worker_softmax_targets[count])
                    model = model.to('cpu')
                    worker_id_list.append(i)
                    count += 1
                worker_softmax_targets = np.array(worker_softmax_targets)
                kmeans = KMeans(n_clusters=args.cluster_num)
                pred = kmeans.fit_predict(worker_softmax_targets)
        else:
            with torch.no_grad():
                worker_softmax_targets = [[] for i in range(len(workers))]
                worker_id_list = []
                count = 0
                for i,model in enumerate(self.models):
                  if model==None:
                    pass
                  else:
                    model = model.to(args.device)
                    model.eval()
                    for data,_ in self.unlabeled_dataloader:
                      data = data.to(args.device)
                      worker_softmax_targets[count].append(model(data).to('cpu').detach().numpy())
                    worker_softmax_targets[count] = np.array(worker_softmax_targets[count])
                    model = model.to('cpu')
                    worker_id_list.append(i)
                    count += 1
                worker_softmax_targets = np.array(worker_softmax_targets)
                kmeans = KMeans(n_clusters=args.cluster_num)
                pred = kmeans.fit_predict(worker_softmax_targets)
            
    self.cluster = []
    for i in range(args.cluster_num):
      self.cluster.append([])
    for i,cls in enumerate(pred):
      self.cluster[cls].append(worker_id_list[i])
    for worker in workers:
      idx = worker_id_list.index(worker.id)
      worker.cluster_num = pred[idx]
        
  def decide_other_model(self,workers):
    for worker in workers:
      cls = worker.cluster_num
      '''if number of worker in cluster is one, other model is decided by random in all workers. '''
      if len(self.cluster[cls])==1:
        while True:
          other_worker = random.choice(workers)
          other_model_id = other_worker.id
          if worker.id!=other_model_id:
            break
      else:
        while True:
          other_model_id = random.choice(self.cluster[cls])
          if worker.id!=other_model_id:
            break
      worker.other_model_id = other_model_id


# In[ ]:


class Worker():
  def __init__(self,i,trainset,valset,testset):
    self.id = i
    self.cluster_num = None
    self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2)
    self.valloader = torch.utils.data.DataLoader(valset,batch_size=args.test_batch,shuffle=False,num_workers=2)
    self.testloader = torch.utils.data.DataLoader(testset,batch_size=args.test_batch,shuffle=False,num_workers=2)
    self.local_model = eval(model_name)
    self.local_model_id = i
    self.other_model = None
    self.other_model_id = None
    self.train_data_num = len(trainset)
    self.test_data_num = len(testset)

  def local_train(self):
    if args.dataset_name=='Sent140':
        self.local_model.train()
        self.other_model.train()
        self.local_model = self.local_model.to(args.device)
        self.other_model = self.other_model.to(args.device)
        local_hidden_train = self.local_model.init_hidden(args.batch_size)
        other_hidden_train = self.other_model.init_hidden(args.batch_size)
        local_optimizer = optim.SGD(self.local_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        other_optimizer = optim.SGD(self.other_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
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
            other_optimizer.zero_grad()
            local_hidden_train = repackage_hidden(local_hidden_train)
            other_hidden_train = repackage_hidden(other_hidden_train)
            local_outputs, local_hidden_train = self.local_model(data, local_hidden_train) 
            other_outputs, other_hidden_train = self.other_model(data, other_hidden_train) 

            #train local_model
            ce_loss = args.criterion_ce(local_outputs.t(), torch.max(labels, 1)[1])
            kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim = 1),F.softmax(Variable(other_outputs), dim=1))
            loss = ce_loss + kl_loss
            running_loss += loss.item()
            _, predicted = torch.max(local_outputs.t(), 1)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
            count += len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)
            local_optimizer.step()

            #train other_model
            ce_loss = args.criterion_ce(other_outputs.t(), torch.max(labels, 1)[1])
            kl_loss = args.criterion_kl(F.log_softmax(other_outputs, dim = 1),F.softmax(Variable(local_outputs), dim=1))
            loss = ce_loss + kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), args.clip)
            other_optimizer.step()

        return 100.0*correct/count,running_loss/len(self.trainloader)
    
    
    else:
        self.local_model = self.local_model.to(args.device)
        self.other_model = self.other_model.to(args.device)
        local_optimizer = optim.SGD(self.local_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        other_optimizer = optim.SGD(self.other_model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
        self.local_model.train()
        self.other_model.train()
        for epoch in range(args.local_epochs):
          running_loss = 0.0
          correct = 0
          count = 0
          for (data,labels) in self.trainloader:
            data,labels = Variable(data),Variable(labels)
            data,labels = data.to(args.device),labels.to(args.device)
            local_optimizer.zero_grad()
            other_optimizer.zero_grad()
            local_outputs = self.local_model(data)
            other_outputs = self.other_model(data)
            #train local_model
            ce_loss = args.criterion_ce(local_outputs,labels)
            kl_loss = args.criterion_kl(F.log_softmax(local_outputs, dim = 1),F.softmax(Variable(other_outputs), dim=1))
            loss = ce_loss + kl_loss
            running_loss += loss.item()
            predicted = torch.argmax(local_outputs,dim=1)
            correct += (predicted==labels).sum().item()
            count += len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), args.clip)
            local_optimizer.step()

            #train other_model
            ce_loss = args.criterion_ce(other_outputs,labels)
            kl_loss = args.criterion_kl(F.log_softmax(other_outputs, dim = 1),F.softmax(Variable(local_outputs), dim=1))
            loss = ce_loss + kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), args.clip)
            other_optimizer.step()

        return 100.0*correct/count,running_loss/len(self.trainloader)        

        
  def validate(self):
    acc,loss = test(self.local_model,args.criterion_ce,self.valloader)
    return acc,loss


  def model_replacement(self):
    _,loss_local = test(self.local_model,args.criterion_ce,self.valloader)
    _,loss_other = test(self.other_model,args.criterion_ce,self.valloader)
    if loss_other<loss_local:
        self.local_model_id = self.other_model_id


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


server = Server(unlabeled_dataset)
workers = server.create_worker(federated_trainset,federated_valset,federated_testset)
acc_train = []
loss_train = []
acc_valid = []
loss_valid = []

early_stopping = Early_Stopping(args.partience)

start = time.time()

for epoch in range(args.global_epochs):
  if epoch in args.turn_of_cluster_num:
    idx = args.turn_of_cluster_num.index(epoch)
    args.cluster_num = args.cluster_list[idx]
  sample_worker = server.sample_worker(workers)
  server.collect_model(sample_worker)
  server.clustering(sample_worker)
  server.decide_other_model(sample_worker)
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
  if epoch in args.turn_of_replacement_model:
    for worker in sample_worker:
      worker.model_replacement()
  server.aggregate_model(sample_worker)
  server.return_model(sample_worker)
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
  worker.local_model = worker.local_model.to(args.device)
  acc_tmp,loss_tmp = test(worker.local_model,args.criterion_ce,worker.testloader)
  acc_test.append(acc_tmp)
  loss_test.append(loss_tmp)
  print('Worker{} accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))
  worker.local_model = worker.local_model.to('cpu')

end = time.time()

acc_test_avg = sum(acc_test)/len(acc_test)
loss_test_avg = sum(loss_test)/len(loss_test)
print('Test  loss:{}  accuracy:{}'.format(loss_test_avg,acc_test_avg))


# In[ ]:


args.local_epochs = 2


# In[ ]:


acc_tune_test = []
loss_tune_test = []
acc_tune_valid = []
loss_tune_valid = []

start = time.time()

for i,worker in enumerate(workers):
    worker.local_model = worker.local_model.to(args.device)
    _,_ = train(worker.local_model,args.criterion_ce,worker.trainloader,args.local_epochs)
    acc_tmp,loss_tmp = test(worker.local_model,args.criterion_ce,worker.valloader)
    acc_tune_valid.append(acc_tmp)
    loss_tune_valid.append(loss_tmp)
    print('Worker{} Valid accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))
    
    acc_tmp,loss_tmp = test(worker.local_model,args.criterion_ce,worker.testloader)
    acc_tune_test.append(acc_tmp)
    loss_tune_test.append(loss_tmp)
    print('Worker{} Test accuracy:{}  loss:{}'.format(i+1,acc_tmp,loss_tmp))
    worker.local_model = worker.local_model.to('cpu')

end = time.time()

acc_valid_avg = sum(acc_tune_valid)/len(acc_tune_valid)
loss_valid_avg = sum(loss_tune_valid)/len(loss_tune_valid)
print('Validation(tune)  loss:{}  accuracy:{}'.format(loss_valid_avg,acc_valid_avg))
acc_test_avg = sum(acc_tune_test)/len(acc_tune_test)
loss_test_avg = sum(loss_tune_test)/len(loss_tune_test)
print('Test(tune)  loss:{}  accuracy:{}'.format(loss_test_avg,acc_test_avg))


# In[ ]:


filename = 'FedMe_{}'.format(args.dataset_name)
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


acc_tune_valid = pd.DataFrame(acc_tune_valid)
loss_tune_valid = pd.DataFrame(loss_tune_valid)
acc_tune_test = pd.DataFrame(acc_tune_test)
loss_tune_test = pd.DataFrame(loss_tune_test)

acc_tune_valid.to_csv(result_path+filename+'_fine-tune_valid_acc.csv',index=False, header=False)
loss_tune_valid.to_csv(result_path+filename+'_fine-tune_valid_loss.csv',index=False, header=False)
acc_tune_test.to_csv(result_path+filename+'_fine-tune_test_acc.csv',index=False, header=False)
loss_tune_test.to_csv(result_path+filename+'_fine-tune_test_loss.csv',index=False, header=False)

