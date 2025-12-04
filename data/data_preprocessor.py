# data/data_preprocessor.py

# Import necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class DataPreprocessor:
    def __init__(self):
        self.dataset = None
    
    def load_hc3_dataset(self):
        # Load the HC3 dataset from the Hugging Face datasets library
        self.dataset = load_dataset("Hello-SimpleAI/chatgpt-detector/HC3", "all")
        return self.dataset

    
    def preprocess(self):
        #convert to binary classification format
        processed_data = []

        for example in self.dataset['train']:
            #Human answer is labeled as 1, GPT answer is labeled as 0

            #Human answer
            if example['human_answers']:
                processed_data.append({
                    'text': example['human_answers'][0],
                    'label': 1,
                    'source': example['question']
                })
        
            #GPT answer
            if example['gpt_answers']:
                processed_data.append({
                    'text': example['gpt_answers'][0],
                    'label': 0,
                    'source': example['question']
                })
        return pd.DataFrame(processed_data)
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        #split data into train, validation, and test sets and save as csv files

        train_df,test_df = train_test_split(df,test_size=test_size, random_state=42)

        train_df,val_df = train_test_split(train_df,test_size=val_size, random_state=42)

        train_df.to_csv('data/processed/train.csv', index=False)
        val_df.to_csv('data/processed/val.csv', index=False)
        test_df.to_csv('data/processed/test.csv', index=False)

        return train_df, val_df, test_df

       