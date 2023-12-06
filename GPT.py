from TransformerComponents import Block

import torch
import random
import torch.nn as nn
from torch.nn import functional as F


class GenerativelyPretrainedTransformer(nn.Module):

    def __init__(self, vocabSize, embeddingDimensions, blockSize, numLayers, numHeads, batchSize, dropout = 0.2):

        super().__init__()

        # ratios for splitting training and testing data
        self.TRAINING_RATIO = 0.9
        self.VALIDATION_RATIO = 0.1

        # number of tokens used as context
        self.BLOCK_SIZE = blockSize
        self.BATCH_SIZE = batchSize # number of sequences processed in parallel

        # testing values
        self.EVAL_ITERATIONS = 200
        self.EVAL_INTERVALS = 300

        # initialize lookup tables for token values and positions in context
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, embeddingDimensions)
        self.positionEmbeddingTable = nn.Embedding(blockSize, embeddingDimensions)

        # transformer initialization
        # TODO:::: ->>> FINISH THIS COMMENT
        # initalize a sequential container of Block objects, info on Blocks can be found 
        self.blocks = nn.Sequential(*[Block(embeddingDimensions, numHeads, blockSize, dropout) for _ in range(numLayers)]) # create numLayer blocks
        self.finalNormLayer = nn.LayerNorm(embeddingDimensions)

        # layer responsible for turning token embeddings into tokens
        self.languageHead = nn.Linear(embeddingDimensions, vocabSize)

        self.apply(self._init_weights)

    
    '''
    Initialize the weights with a normal distribution and biases as zero
    if the model chooses to use them.
    '''
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    '''
    Forward pass through the model
    '''
    def forward(self, input, targets=None):
        # allow for flexible input
        B, T = input.shape

        # create list of position values 
        indexVals = torch.arange((T))

        # get the trained embeddings for the input tokens
        tokenEmbeddings = self.tokenEmbeddingTable(input) # size should be (B, T, C) -> C = embedding dimensions
        positionEmbeddings = self.positionEmbeddingTable(indexVals) # values of embeddings for each position in context -> size should be (batchsize, embeddingDims)

        # create embedding table which accounts for context position and token value
        combinedEmbeddings = tokenEmbeddings + positionEmbeddings

        # feed combined embeddings into a block to determine the resulting embeddings
        combinedEmbeddings = self.blocks(combinedEmbeddings)

        # normalize each token embeddings
        combinedEmbeddings = self.finalNormLayer(combinedEmbeddings)

        # pass through layer which converts embeddings to tokens using the combined embeddings
        logits = self.languageHead(combinedEmbeddings) # size should be (B, T, vocabSize) 

        if targets is None:
            # no known targets, cant calculate loss
            loss = None
        else:
            # calculate loss of target

            # need to convert logits to 2d
            # rows = value of next token activation for next row
            # cols = value at each position in the context 
            (numBatches, contextLength, numVocab) = logits.shape
            logits = logits.view(numBatches * contextLength, numVocab)

            # conform tagets to 1 dimension
            targets = targets.view(numBatches * contextLength) # expected next charater at each context length for each sample

            # calculate loss as the negative log likelihood of preditctions associated to the target
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    '''
    Expected input: (B, T) -> batches of tokens in a sequence
    '''
    def generate_text(self, input, numTokens):

        # index is the (batch, context length) array of indices for the current context
        for i in range(numTokens):

            # crop input to first blocksize tokens
            croppedInput = input[:, -self.BLOCK_SIZE:]
            
            # get predicted activations of next token at each position in the sample
            logits, loss = self(croppedInput)

            # only want the logits for the last token in the block
            finalLogits = logits[:, -1, :] # logits for prediciton of next token

            # use PyTorch softmax to get a probability distribution of next token based on the logits 
            probDist = F.softmax(finalLogits, dim=-1) # apply across z axis

            # sample next token from probability distibution
            nextToken = torch.multinomial(probDist, num_samples=1)

            # add the next token to the input so it is used as context
            input = torch.cat((input, nextToken), dim=1) 

        return input
    

