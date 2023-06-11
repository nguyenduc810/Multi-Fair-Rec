import numpy as np

# import theano
# import theano.tensor as T
import keras
from keras import backend as K
from keras.initializers import RandomNormal
from keras.regularizers import l2  #, activity_l2
from keras.models import  Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Concatenate, Reshape, Multiply, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from time import time
import sys
import argparse
import multiprocessing as mp

import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F


class MLP(nn.Module): 
    def __init__(self, num_users, num_items, layers = [100,256], batch_size = 1024) :
        super(MLP, self).__init__()
        #assert len(layers) == len(reg_layers)
        # self.num_layer = len(layers) #Number of layers in the MLP
        # # Input variables
        # self.user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
        # self.item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(layers)
        self.batch_size = batch_size
        
        self.user_embeddings = nn.Embedding(num_users,layers[0]//2).to(torch.device('cuda'))
        self.item_embeddings = nn.Embedding(num_items,layers[0]//2).to(torch.device('cuda'))
        self.layer = nn.Linear(layers[0],layers[1]).to(torch.device('cuda'))
        # self.layer_2 = nn.Linear(layers[0], layers[1]).to(torch.device('cuda'))
        self.linear_layer  = nn.Linear(layers[1], 1).to(torch.device('cuda'))

        self.user_embeddings.weight.data = torch.nn.init.normal_(self.user_embeddings.weight.data, 0, 0.01)
        self.item_embeddings.weight.data = torch.nn.init.normal_(self.item_embeddings.weight.data, 0, 0.01)
        self.layer.weight.data = torch.nn.init.normal_(self.layer.weight.data,0,0.01)
        #self.layer_2.weight.data = torch.nn.init.normal_(self.layer_2.weight.data,0,0.01)
        self.linear_layer.weight.data = torch.nn.init.normal_(self.linear_layer.weight.data,0,0.01)

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight, self.layer.weight, self.linear_layer.weight]
        self.myparameters_2 = [self.layer, self.linear_layer]
    
    def forward (self, user_id, pos_id, neg_id):
        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        neg_emb = self.item_embeddings(neg_id)

        user_latent = torch.reshape(user_emb, (self.batch_size,-1))
       # print(user_latent.shape)
        item_pos = torch.reshape(pos_emb, (self.batch_size, -1))
        #print(item_pos.shape)
        item_neg = torch.reshape(neg_emb, (self.batch_size, -1))

        vector_pos = torch.cat((user_latent, item_pos), dim=1)
        #print(vector_pos.shape)
        vector_neg = torch.cat((user_latent, item_neg), dim=1)

        #for idx in range(0, self.num_layer -1):
            #layer = Dense(self.layers[idx], kernel_regularizer=l2(self.reg_layers[idx]), activation='relu', name='layer%d' % idx)
        activation = nn.ReLU()
        vector_pos = activation(self.layer(vector_pos))
        vector_neg = activation(self.layer(vector_neg))


        # layer = nn.Linear(self.layers[idx])
            # vector = layer(vector)
        sigmoid = nn.Sigmoid()
        pos_scores = sigmoid(self.linear_layer(vector_pos))
        #print(pos_scores)
        neg_scores = sigmoid(self.linear_layer(vector_neg))

        tmp = pos_scores - neg_scores

        maxi = nn.LogSigmoid()(tmp)
        bpr_loss = -torch.sum(maxi)

        return bpr_loss

    def predict(self, user_id):
      #print(user_id.shape[0])
      user_emb  = self.user_embeddings(user_id)
      user_emb  = user_emb.repeat_interleave(self.num_items, dim = 0)
      item_emb = self.item_embeddings.weight.repeat(user_id.shape[0],1)

      # print(user_emb.shape)
      # print(item_emb.shape)
      
      vector = torch.cat((user_emb, item_emb), dim = 1)

      activation = nn.ReLU()
      vector = activation(self.layer(vector))
      
      sigmoid = nn.Sigmoid()
      scores_all = sigmoid(self.linear_layer(vector))
      pred  = torch.reshape(scores_all, (user_id.shape[0], self.num_items))

      return pred