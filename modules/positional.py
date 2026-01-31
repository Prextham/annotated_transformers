import torch
import torch.nn as nn
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        
        #d_model basically tells us the total number of dimensions in this vector(512D in our case)
        #Usage of dropout: We use dropout in this case, to prevent overfitting, that is we dont want,
        #our model to look at a particular word and assume it exists at that particular position
        #So we drop some number from our vectors so model knows the word too and not memorizing the position
        
        #max_len we set it because we dont want our model to waste space
        #Its a killer in self attention mechanism, as every word looks at every other word
        #Lets say we have 5000 words, 5000x5000=25000000 connections, which becomes too big
        #Imagine a document with 100000 words, it becomes really big
        #We will run into out of memory issues if the length is too big
        
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        
        #max_len is the number of rows
        #d_model or dimensions is the number of columns 
        #Instead of loops which would be time and memory consuming, we will create position and frequency matrix, multiply them to create the grid
        
        pos=torch.arange(0,max_len).unsqueeze(1)
        
        #arange function to get an array values [0,1,...511]
        #To convert this flat horizontal array line, into individual rows so we can multiply each value with frequency, we use unsqueeze function
        #unsqueeze(1) converts this [0,1,...511] to this [[0],[1],[2]....,[511]]
        #Every item is a separate row
        
        div_term=torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))
        
        #torch.arange generates us an array with even values [0,2,4,....,d_model]
        #convert this into float value as we will be multiplying this with a float value, to prevent mismatch
        #with this we have created frequency logic 
        
        pe[:,0::2]=torch.sin(pos*div_term)
        pe[:,1::2]=torch.cos(pos*div_term)
        
        #we use slicing instead of creating another variable
        #if we use for loop instead of slicing, its executed by the cpu but the commands are sent to the gpu one by one
        #sending 256 small commands to do one task, is time wasting, gpu waits more than actually doing the work
        
        
        pe=pe.unsqueeze(0)
        
        #In deep learning, we usually never process one sentence at a time, we process batches
        #we expect data in the form of (batch size, sequence length, dimensions of model)
        #(1,512,512) is the shape right after unsqueeze(0)
        
        
        self.register_buffer('pe',pe)
        
        #tpyical problem that might arise when we try to move this matrix from cpu to gpu
        
        