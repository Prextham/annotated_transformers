import torch 
import torch.nn as nn
import math 

class Embeddings(nn.Module):
    def __init__(self,d_model, vocab):
        super().__init__()   #Turning on the brain infrastructure of this class
        self.lut=nn.Embedding(num_embeddings=vocab,embedding_dim=d_model) 
        #Lookup table for this class
        #Number of embeddings is the size of vocabulary
        #Dimension size of embedding matrix
        
        self.d_model=d_model #Storing the dimension size as well for future refernces
        
        
    def forward(self,x):
        #The value x contains both batch_size and sequence_length as shape Ex.x=[2,3]
        #batch_size tells us the number of sentences/sequences involved (2sentences)
        #sequence_length tells us how many words/tokens are present in each sequence (3 words in each sentence)
        
        return self.lut(x)*math.sqrt(self.d_model)
    
        #With the above formula, we basically crank up the value of our word vector, before adding 
        #positional encoding value to it(ranges from -1 to 1) 
        #With this we wont lose the semantic meaning of our word when PE is added
        