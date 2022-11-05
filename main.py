import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
import torch.nn as nn
from tqdm.notebook import tqdm


from models import essay_model
from config import config


def training_function(dataloader, model, device, optimizer):
    
    model.train()
    loss_sum = 0.
    total = 0
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, data in iterator:
        ids = data['ids']
        token_type_ids = data['token_type_ids']
        mask = data['mask']
        targets = data['target']
#         print("ids:", ids)
#         print("tt ids:",token_type_ids)
#         print("mask:", mask)
#         print("target:",targets)
        
        ids = ids.squeeze().to(device, dtype=torch.long)
        token_type_ids = token_type_ids.squeeze().to(device, dtype=torch.long)
        mask = mask.squeeze().to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids, token_type_ids, mask)
        
#         print(outputs.size())
#         print(targets.size())
        
        loss = config.loss_fn(outputs, targets)
        loss_sum += loss.item()
        
        loss.backward()
        optimizer.step()
        total += targets.shape[0]
        
        del ids, token_type_ids, mask, targets, outputs, loss
        
    return loss_sum/total



def validation_function(dataloader, model, device):
    
    model.eval()
    loss_sum = 0.
    total = 0
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, data in iterator:
        ids = data['ids']
        token_type_ids = data['token_type_ids']
        mask = data['mask']
        targets = data['target']
        
        ids = ids.squeeze().to(device, dtype=torch.long)
        token_type_ids = token_type_ids.squeeze().to(device, dtype=torch.long)
        mask = mask.squeeze().to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        with torch.no_grad():
            outputs = model(ids, token_type_ids, mask)
        
        loss = config.loss_fn(outputs, targets)
        loss_sum += loss.item()
        total += targets.shape[0]
        
        del ids, token_type_ids, mask, targets, outputs, loss
        
    return loss_sum/total



def testing_function(dataloader, model, device):
    
    predictions = []
    model.eval()
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, data in iterator:
        ids = data['ids']
        token_type_ids = data['token_type_ids']
        mask = data['mask']
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(ids, token_type_ids, mask)
        predictions.append(outputs.detach().cpu())
    
    preds = np.concatenate(predictions)
    return preds

device = config.device

loss_training = []
loss_validation = []

train_data, valid_data = train_test_split(train, test_size=0.2, random_state=77, shuffle=True)
train_data, valid_data = train_data.reset_index(), valid_data.reset_index()

model = essay_model(config)
model.to(device)
train_dataset = train_valid_dataset(train_data)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.train_batch_size)
valid_dataset = train_valid_dataset(valid_data)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.valid_batch_size)
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0},]
optimizer = AdamW(optimizer_parameters, lr=0.00001)
    
for epoch in range(config.epochs):
    print("#################### EPOCH: %d ####################" % (epoch+1))
    train_loss = training_function(train_dataloader, model, device, optimizer)
    valid_loss = validation_function(valid_dataloader, model, device)
    loss_training.append(train_loss)
    loss_validation.append(valid_loss)
    print("Training Loss: %f, Validation Loss: %f" % (train_loss, valid_loss))

testing_dataset = test_dataset(test)
test_dataloader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=config.valid_batch_size)
predictions = testing_function(test_dataloader, model, device)