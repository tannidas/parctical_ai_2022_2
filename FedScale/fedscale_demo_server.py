#!/usr/bin/env python
# coding: utf-8

# # Federated Learning for Image Classification using Fedscale

# ## Server Side

# In[1]:


import sys, os

from fedscale.core.execution.client import Client
from fedscale.core.aggregation.aggregator import Aggregator
## import fedscale.core.logger.execution as args
from fedscale.core.config_parser import args


Demo_Aggregator = Aggregator(args)

### On CPU
args.use_cuda = "True"  ## originally False
Demo_Aggregator.run()


# In[ ]:


get_ipython().system('tensorboard --logdir=./logs/demo_job --port=6007 --bind_all')

