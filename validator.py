# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:12:29 2023

@author: Chovatiya
"""
import torch
import torch.nn.functional as F
DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def accuracy(out, targets):
    _, preds = torch.max(out, dim=1)
    acc = torch.tensor(torch.sum(preds==targets).item() / len(preds))
    return acc

@torch.no_grad()
def evaluate(model, val_dl):
    outputs = []
    model.eval()
    for images, targets in val_dl:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        out = model(images)
        loss = F.cross_entropy(out, targets.view(-1))
        acc = accuracy(out, targets.view(-1))
        output = {'val_loss': loss.detach(), 'val_acc': acc}
        outputs.append(output)
    batch_loss = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_loss).mean()
    batch_acc = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_acc).mean()
    val_dict = {'val_loss': epoch_loss.item(), 'val_acc':epoch_acc.item()}
    
    return val_dict