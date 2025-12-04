import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class AITextDetector(nn.Module):
    def __init__(self,model_name='distilbert-base-uncased',dropout_rate=0.1):
        super().__init__()

        #load pre-trained DistilBERT model
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.config = DistilBertConfig.from_pretrained(model_name)

        #classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, 2) 

        #confidence calibration layer
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask):
        #get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        #use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:,0,:]
        pooled_output = self.dropout(pooled_output)

        #classification logits
        logits = self.classifier(pooled_output)

        return logits
    
    def predict_with_confidence(self,input_ids,attention_mask):
        #return predictions with calibrated confidence scores
        with torch.no_grad():
            logits = self.forward(input_ids,attention_mask)

            #apply temperature scaling for confidence calibration
            scaled_logits = logits / self.temperature

            #get probabilities
            probabilities = nn.Softmax(dim=-1)(scaled_logits)
            confidence, predictions = torch.max(probabilities,dim=-1)

            #convert to percentages
            confidence_percent = confidence.item() * 100.0

        return predictions.item(), confidence_percent

