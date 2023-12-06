import torch
import random
import torch.nn as nn
from torch.nn import functional as F

from GPT import GenerativelyPretrainedTransformer

class Model:

    def __init__(self, content, vocabSize, blockSize, embeddingDimensions, numLayers, numHeads, batchSize, learningRate, trainingIters):

        # ratios for splitting training and testing data
        self.TRAINING_RATIO = 0.9
        self.VALIDATION_RATIO = 0.1
        
        self.BLOCK_SIZE = blockSize # number of tokens used as context ro guess next one
        self.BATCH_SIZE = batchSize # number of sequences processed in parallel
        self.NUM_LAYERS = numLayers # number of layers in transformer
        self.NUM_HEADS = numHeads # Number of heads in the linear layer
        self.EMBEDDING_DIMS = embeddingDimensions # number of dimensions to embed each token into
        self.LEARNING_RATE = learningRate #set learning rate
        self.TRAINING_ITERATIONS = trainingIters # number of training iterations to go through

        # testing values
        self.EVAL_ITERATIONS = 200
        self.EVAL_INTERVALS = 300

        # split input content into training and validation sets
        contentLength = len(content)
        lengthTraining = int(contentLength * self.TRAINING_RATIO)
        self.trainingData = content[:lengthTraining]
        self.validationData = content[lengthTraining:]
        
        # initalize the transfomer
        self.m = GenerativelyPretrainedTransformer(vocabSize, self.EMBEDDING_DIMS, self.BLOCK_SIZE, self.NUM_LAYERS, self.NUM_HEADS, self.BATCH_SIZE, 0.2)
        
        # advanced optimzer from Pytorch
        self.optimizer = torch.optim.AdamW(self.m.parameters(), lr = self.LEARNING_RATE)
        

    '''
    Helper function to create a batch of data. A batch of data is essentially
    a sequence of characters picked at random within the data, the sequence is of 
    the determined context length that the model uses. 
    '''
    def create_sample_batch(self, data):

        # convert data into tensor
        data = torch.tensor(data)

        # use PyTorch to create BATCH_SIZE integers that contain valid starting indicies for sample block
        startingIndicies = torch.randint(len(data) - self.BLOCK_SIZE, (self.BATCH_SIZE,))

        batchContext = torch.stack([data[i: i + self.BLOCK_SIZE] for i in startingIndicies])
        batchTargets = torch.stack([data[i + 1: i + self.BLOCK_SIZE + 1] for i in startingIndicies])

        # batchContext and targets are batchsize x blocksize matrices
        # rows = values for a sample in the batch 
        # columns = distance to look ahead
        # for context of 3 in sample 0, the context is batchContext[0][:3] and target is batchTargets[0][3]
        return batchContext, batchTargets


    '''
    Function to run a predetermined number of batches of model sampling
    using both training and validation data. The function returns the mean 
    loss found for each data source
    '''
    @torch.no_grad() # do not store gradients in this calculation 
    def calculate_loss(self):

        # average loss for training data and validation data
        averageLosses = []

        self.m.eval() # set model to evaluation phase
        for stage in ['training', 'validation']:

            if stage == 'training':
                sampleData = self.trainingData # sampling loss found in training
            else:
                sampleData = self.validationData # sampling loss found in validation

            # intialize every evaluation to have value of zero for stage
            losses = torch.zeros(self.EVAL_INTERVALS)  

            # calculate loss for many iterations
            for i in range(self.EVAL_ITERATIONS):
                batchInput, batchoutput = self.create_sample_batch(sampleData)
                _, loss = self.m(batchInput, batchoutput)
                losses[i] = loss
            
            # represent loss as mean of all samples
            averageLosses.append(losses.mean())
        
        # reset model into training phase
        self.m.train()
        return averageLosses


    '''
    Based on a predetermined number of iterations, run a forward pass
    and back propegate on the results in order to train parameters to 
    compute a more accurate prediction
    '''
    def train_model(self):

        # run give amount of times
        for _ in range(self.TRAINING_ITERATIONS):

            # sample a batch of data
            currentContext, currentTarget = self.create_sample_batch(self.trainingData)

            # pass batch through model -> evaluate loss
            _, loss = self.m(currentContext, currentTarget)

            # reset gradients for optimization
            self.optimizer.zero_grad(set_to_none=True)

            # back propegate loss in order to train model to get correct prediction embeddings through the model 
            loss.backward()

            # ajdust PyTorch optimzer
            self.optimizer.step()



    '''
    Method to create text of a given length. Initally, the given context is an empty character,
    although this can be possibly changed to take user input.
    '''
    def create(self, length):

        initalContext = torch.zeros((1, 1), dtype = torch.long) # inital context is 1 sample of 1 character which is an empty char

        return self.m.generate_text(initalContext, length)[0].tolist()
    
    