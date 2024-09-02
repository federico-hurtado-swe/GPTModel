# GPTModel
GPT model developed as a final project for Intro to AI class. 

## Prerequisites:
- pip install tiktoken
- install pytorch: https://pytorch.org/get-started/locally/ 

## Project Description
This is a basic Generative Pretrained Transformer model that I created for my Intro To AI final project at Virginia Tech. I completed this project individually, and made use of many
online resources in order to learn how to get my first experience with building a real neural network from scratch. I hope to build off these skills in future work.

The model uses character based encoding to predict the next character in a sequence after using previous characters as context. The number of characters to use as context is an important
parameter to set, however it needs to be determined based on CPU quality; I have found a large context length is very computationally expensive. 

The output of the model is set to create short snippets of Family Guy scripts, however if given any other text input it should be able to recreate it with generally high accuracy.

## Usage
1. Install all prerequistes
2. Run ./driver.py
3. Wait for the model to be created, and .txt files will be created in a directory called TrainingResults where the model output every 500 training iterations will print out. NOTE: This may take a long time depending on hardware. For reference, it takes roughly 30 minutes to run on my Windows XPS laptop.
