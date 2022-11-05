from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import transformers
from transformers import get_linear_schedule_with_warmup, AdamW, get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

class essay_model(nn.Module):
    
    def __init__(self, config, num_labels=6):
        super(essay_model, self).__init__()
        self.bert_model = transformers.AutoModel.from_pretrained(config.model)
        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 64)
        self.fc2 = nn.Linear(64, num_labels)
    
    def forward(self, ids, token_type_ids, mask):
        _, outputs = self.bert_model(ids, token_type_ids, mask, return_dict=False)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs