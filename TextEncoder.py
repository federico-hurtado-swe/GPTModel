# import nltk
# nltk.download('punkt')

import json
import tiktoken
from nltk.tokenize import word_tokenize

''''
Character based encoder: This encoder will parse a given input string at initialization and keep track of all
found characters. It will sort the given characters and tokenize them in alphabetical order. 
Has the capability to encode/decode at a character level.
'''
class CharacterEncoder:

    '''
    Initialization: find all possible characters and number of possible characters
    '''
    def __init__(self, inputString):
        self.possibleChars = sorted(list(set(''.join(inputString)))) # store all possible characters sorted
        self.numTokens = len(self.possibleChars)


    '''
    Encodes characters into integer tokens based on alphabetical order of numbers found.
    Given an input string, it will return an array of integers where each integer represents a chatacter
    '''
    def encode(self, inputString):

        tokenizedChars = []

        for char in inputString:
            token = self.character_to_token(char)
            tokenizedChars.append(token)
        
        return tokenizedChars
    
    ''''
    Helper method to convert a character into an integer based on found characters
    '''
    def character_to_token(self, character):
        for index, char in enumerate(self.possibleChars):
            if char == character:
                return index


    '''
    Helper method to decode tokens (integers) back into characters from the 
    found list of characters earlier
    '''
    def decode_tokens(self, tokens):
        
        decodedChars = []

        for token in tokens:
            decodedChar = self.possibleChars[token]
            decodedChars.append(decodedChar)
        
        return decodedChars

    '''
    Returns the number of unique tokens found in the dataset
    '''
    def num_vocab(self):
        return self.numTokens
    


class TikTokenEncoder:

    '''
    Initialize the tiktoken encoding version and model to the given request
    Keep track of vocab count as well
    '''
    def __init__(self, encodingVersion, encoding_model):
        tiktoken.get_encoding(encodingVersion)
        self.encoding = tiktoken.encoding_for_model(encoding_model)
        self.vocabCount = self.encoding.n_vocab

    '''
    Use tiktoken to encode content according to set standards
    '''
    def encode_content(self, content):
        encodedContent = self.encoding.encode(content)
        return encodedContent
    
    '''
    Use tiktoken to decode content according to set standards
    '''
    def decode_tokens(self, tokens):
        decodedContent = self.encoding.decode(tokens)
        return decodedContent

    ''''
    Getter method for the number of tokens possibe for the given encoding type
    '''
    def num_tokens(self):
        return self.vocabCount
    

class WordEncoder:

    '''
    Initialize a word encoder from the data stored in a JSON file created at launch
    '''
    def __init__(self):
        jsonFilePath = "shakespeare_json.json" # possibly pass this on thru constructor when choosing training data in the future
        self.possibleWords, self.numPossibleWords = self.read_json_data(jsonFilePath)

        
    '''
    Helper method to read word data from JSON file created upon preprocessing. Info on this can be found in
    driver.py
    '''
    def read_json_data(self, jsonFilePath):
        try:
            with open(jsonFilePath, 'r') as jsonFile:
                data = json.load(jsonFile)

                # extract needed info
                length = data["Number of Words"]
                uniqueWords = data["Words"]
                return uniqueWords, length
        except:
            print("JSON data not found: program does not support word tokenization until this is resolved. Please restart program.")
            return None
        

    '''
    Pre: words in input string MUST be found within training data at least once.
    '''
    def encode(self, inputString):

        tokenizedWords = []

        for word in inputString.split():
            token = self.word_to_token(word)
            tokenizedWords.append(token)

        return tokenizedWords

    ''''
    Helper method to convert a character into an integer based on found characters
    '''
    def word_to_token(self, inputWord):
        for index, word in enumerate(self.possibleWords):
            if inputWord == word:
                return index


    '''
    Decodes a token into a word. Token is the index of the wanted word within
    the possible_words list
    '''
    def decode_tokens(self, tokens):
        decodedTokens = []

        for token in tokens:
            decodedWord = self.possibleWords[token]
            decodedTokens.append(decodedWord)

        return decodedTokens

    '''
    Returns the number of possible words in the data set. For model to work correctly, all words
    in the input must be in this data set, and the model can only output words from this data set.
    '''
    def num_vocab(self):
        return self.numPossibleWords