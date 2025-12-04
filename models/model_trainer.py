import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import yaml

class ModelTrainer:
    def __init__(self,config_path='config/hyperparameters.yaml'):
        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_epoch(self,model,dataloader,optimizer,scheduler):
        model.train()
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = model(input_ids,attention_mask)
            loss = criterion(outputs,labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def save_checkpoint(self,model,epoch,path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, path)