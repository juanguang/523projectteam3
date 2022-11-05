import torch
import torch.nn as nn


class config():
    model = '../input/bert-base-uncased'
    'max_length' = 512
    'train_batch_size' = 8
    'valid_batch_size' = 8
    'epochs' = 5
    'loss_fn' = nn.SmoothL1Loss()
    'device' = 'cuda' if torch.cuda.is_available() else 'cpu'
    'target_classes' = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
