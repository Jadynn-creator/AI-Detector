# src/dataset.py

#get data from internet
from datasets import load_dataset

def prepare_data():
    #download the dataset
    dataset = load_dataset("Hello-SimpleAI/chatgpt-detector/HC3","all")

    #extract human and AI texts from the first 1000 entries in the training set
    human_texts = [item['human_answers'][0] for item in dataset['train'][:1000]]
    ai_texts = [item['gpt_answers'][0] for item in dataset['train'][:1000]]

    labels = [0]*1000 + [1]*1000  # 0 for human, 1 for AI
