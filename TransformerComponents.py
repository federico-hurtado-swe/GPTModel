import torch
import random
import torch.nn as nn
from torch.nn import functional as F


'''
Building block of layers. Contain 4 layers within it, and 
'''
class Block(nn.Module):

    def __init__(self, embeddingDims, numHeads, blockSize, dropout):
        super().__init__()

        headSize = embeddingDims // numHeads
        self.selfAttention = MultiHeadAttention(embeddingDims, numHeads, headSize, blockSize, dropout) # communication
        self.feedForward = FeedForward(embeddingDims, dropout) # computation

        # Layer normalization for normalizing features in every token before and after attention and feed-forward
        self.layerNorm1 = nn.LayerNorm(embeddingDims) 
        self.layerNorm2 = nn.LayerNorm(embeddingDims) 


    # calculate self attention and feed forward each token
    def forward(self, input):
        # normalize input layers before attention and feeding forward
        input = input + self.selfAttention(self.layerNorm1(input)) # apply residual connection from previous layers
        input = input + self.feedForward(self.layerNorm2(input)) # apply residual connection from previous layers
        return input

'''
Layer to process output from one attention layer in order to better fit
the input of the next attention layer. Applies weights and an activation
function (ReLU) to each token to prepare for next layer.
'''
class FeedForward(nn.Module):

    def __init__(self, embeddingDims, dropout) -> None:
        super().__init__()

        # create simple multilayer perceptron to apply an activation function in sequence
        self.net = nn.Sequential(
            nn.Linear(embeddingDims, 4 * embeddingDims), # project for activation function 
            nn.ReLU(), # activation function
            nn.Linear(4 * embeddingDims, embeddingDims), # projection layer for self attention input
            nn.Dropout(dropout) # dropout some samples to avoid overfitting
        )


    # pass input through net to prepare for next layer
    def forward(self, input):
       
        return self.net(input)
    

'''
Multiple heads of attention in parallel. Self attention refers to using data in 
a batch's own context for training.
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, embeddingDims, numHeads, headSize, blockSize, dropout):
        super().__init__()

        # create many self attention heads to use in parallel
        self.heads = nn.ModuleList([Head(headSize, embeddingDims, blockSize, dropout) for _ in range(numHeads)])

        # projection layer for normalization
        self.projection = nn.Linear(headSize * numHeads, embeddingDims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # concatenate output of all heads over the channel 
        out = torch.cat([currentHead(input) for currentHead in self.heads], dim=-1)

        # apply projection to outcome of multihead attentions and drop some samples to avoid overfitting
        out = self.dropout(self.projection(out))

        return out

'''
Single head of self-attention, to be used in parallel within MultiHeadAttention
'''
class Head(nn.Module):

    def __init__(self, headSize, embeddingDims, blockSize, dropout):
        super().__init__()

        # linear projection to apply for keys
        self.key = nn.Linear(embeddingDims, headSize, bias=False) 
        # linear projection for context locations
        self.query = nn.Linear(embeddingDims, headSize, bias=False)
        # stored weights for found affinities between keys and context locations
        self.value = nn.Linear(embeddingDims, headSize, bias=False)

        # dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # reverse triangular buffer to help normalize activations for each context index
        self.register_buffer('tril', torch.tril(torch.ones(blockSize, blockSize)))
    
    '''
    Expected input:
    [NumBatches, TokensInBatch, tokenEmbeddings]
    '''
    def forward(self, input):
        B, T, C = input.shape
        # index b = batch index
        # index t = token index (token in current batch)
        # index c = embeddings at point in sequence (info in batch i at token j)

        # compute affinity -> how much attention to give each token previous context 
        # affinity = based on key (current token value), determine which previous tokens to give most attention to
        k = self.key(input) # (B, T, C)
        q = self.query(input) # (B, T, C)

        # calculate the affinity values for each index at each token
        affinity = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # calulate affinity and then scale them down to have std dev near 1 -> size = (B, T, T) 
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # get normalized values for weight of context indexes at each token 
        normalizedAffinites = F.softmax(affinity, dim=-1)
        normalizedAffinites = self.dropout(normalizedAffinites)

        # scale with trained weights
        affinityWeights = self.value(input)
        out = normalizedAffinites @ affinityWeights
        return out

