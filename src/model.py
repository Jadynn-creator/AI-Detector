#import necessary libraries
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

class simpleDetector(nn.Module):
    #load pre-trained DistilBERT
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2  # 2 classes: human or AI
        )

    def predict(self,text):
        predicted_class = "Human"

        confidence=85.3

        return predicted_class, confidence