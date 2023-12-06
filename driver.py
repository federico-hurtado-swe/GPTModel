from Model import Model
from TextEncoder import CharacterEncoder, TikTokenEncoder, WordEncoder
import json
import os
import sys

''''
Helper method to process a .txt file and returns a set of unique characters from 
the file
'''
def get_all_unique_words(textFilePath):

    unique_words = set()

    with open(textFilePath, 'r') as file:
        for line in file:
            words = line.split() 
            words = [word.lower() for word in words] # lowercase to increase efficiency by decreasing data size
            words.append('\n') # treat new lines as their own 
            unique_words.update(words)
    return sorted(unique_words)


''''
This step is done to ensure the ability of word tokenization. The purpose of this is to create a JSON file
that holds the unique words in every set of training data. The reason this is necessary is because I struggled to
find a libarary that was not overly big for my machine, yet able to tokenize every single word. 

By doing this I am accepting some tradeoffs:
- Inital preprocessing step which is timely
- Any generated word will be limited to words that are pre-existing in the training data -> this means I will never have 0 Loss 
- I am guaranteed to find an input and output neuron for any found word during training and testing
- I do not need to store words that are never/almost never used 

In the future, I hope to be able to find more creative solutions to my memory space, however the purpose of this
project is to build language models from the ground up instead of utilizing python libraries extensively.

NOTE: In order to use word-based processing, the encoding version needs to be changed in the driver
'''
def preprocessing():
    textFilePath = 'familyguy.txt' # raw text data
    jsonFilePath = "familyguy_json.json" # JSON file holding set of words in data

    # if the json file is not present in the directory, process .txt file to create it before running models
    if not os.path.exists(jsonFilePath):
        
        # get set and count of unique words to put into JSON file
        uniqueWords = get_all_unique_words(textFilePath)
        numUniqueWords = len(uniqueWords)

        jsonData = {
            "Number of Words": numUniqueWords,
            "Words": uniqueWords
        }


        # write it into a JSON file
        with open(jsonFilePath, 'w') as jsonFile:
            json.dump(jsonData, jsonFile)


'''
Helper method initially called to read text files that are going to be used
to train the generative model.

Returns the read file or None if there is an error caught during opening/reading.
'''
def read_training_data():

    # future TODO: add links to multiple texts to download so that user does not need .txt file in their directory
    trainingData = "familyguy.txt"

    try:
        file = open(trainingData, mode='r', encoding='utf-8')
        words = file.read()
    except: # return none on error
        return None
    return words

''''
Helper method to encode content into integer tokens by character or word
'''
def encodeTrainingContent(encodingType, content):

    # values to initialize and return
    encoder = None # encoder created
    encodedContent = None # string encoded into tokens with created encoder
    numVocab = None # 
    encodingUsed = ""

    if encodingType == 0: # char

        encoder = CharacterEncoder(content)
        encodedContent = encoder.encode(content)
        numVocab = encoder.num_vocab()
        encodingUsed = "Character"

    if encodingType == 1: # word

        encoder = WordEncoder() 
        encodedContent = encoder.encode(content.lower()) # lowercase to reduce word count
        numVocab = encoder.num_vocab()
        encodingUsed = "Word"

    # if encodingType == 2: # tiktoken -> not currently supported
    #     encoder = TikTokenEncoder("gpt2", "gpt2")
    #     encodedContent = encoder.encode_content(content)
    #     numVocab = encoder.num_tokens()
    #     encodingUsed = "TikToken"

    return (encoder, encodedContent, numVocab, encodingUsed)



if __name__ == "__main__":

    # read training data
    content = read_training_data()

    # ensure there are no errors reading data
    if content == None: #TODO -> add better check to ensure correct reading
        print("Error reading training data, program is unable to run. Exiting Now.")
        sys.exit() # exit program
        
    # preprocess words into a json file
    preprocessing()

    (currEncoder, encodedContent, numVocab, encodingUsed) = encodeTrainingContent(0, content)
    
    content = encodedContent
    vocabSize = numVocab 
    blockSize = 35 # look at previous 35 characters
    embeddingDimensions = 225 # each character will be embedded into 225 dimensions
    numLayers = 6 # 6 layers
    numHeads = 8 # 8 heads in linear layer
    batchSize = 64 # 32 good for chars
    learningRate = 0.0001 #
    trainingIterations = 500 # create .txt file every 500 iterations


    # create model
    model = Model(content, vocabSize, blockSize, embeddingDimensions, numLayers, numHeads, batchSize, learningRate, trainingIterations)


    ## TRAINING MODEL AND CREATING OUTPUT ##

    fileName = './TrainingResults/ScaledUpModelOutputIteration'

    # create text file with output for no training
    loss = model.calculate_loss()
    output = ''.join(currEncoder.decode_tokens(model.create(100)))
    untrainedFileName = f'{fileName}_0.txt'

   
    # Save untrained output to a file
    with open(untrainedFileName, 'w') as file:
        file.write(f'Loss: {loss}\n\n')
        file.write(output)


    # create txt file for every output to see how training iterations affect output/loss
    for i in range(20):
        # create new file for results and write 200 characters from the trained model
        currFileName = f'{fileName}_{i + 1}.txt'
        model.train_model()
        loss = model.calculate_loss()
        output = ''.join(currEncoder.decode_tokens(model.create(200)))

        with open(currFileName, 'w') as file:
            file.write(f'Iterations trained: {trainingIterations * (i + 1)}\n')
            file.write(f'Loss (training, validation): {loss}\n\n')
            file.write(output)

    

