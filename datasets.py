from config import config

class train_valid_dataset():
    
    def __init__(self, dataset):
        self.data = dataset
        self.classes = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
        self.maxlen = config.max_length
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        full_text = str(self.data['full_text'][index])
        
        tokenizer_dict = tokenizer(full_text,
                                   None,
                                   add_special_tokens=True,
                                   max_length=self.maxlen,
                                   truncation=True,
                                   padding='max_length'
        )
        
        ids = tokenizer_dict['input_ids']
        token_type_ids = tokenizer_dict['token_type_ids']
        attention_mask = tokenizer_dict['attention_mask']
        label = self.data.loc[index, self.classes].to_list()
        
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'mask': torch.tensor(attention_mask, dtype=torch.long),
                'target': torch.tensor(label, dtype=torch.float)
        }



class test_dataset():
    
    def __init__(self, dataset):
        self.config = config
        self.tokenizer = tokenizer
        self.maxlen = config.max_length
        self.essay = dataset['full_text'].values
    
    def __len__(self):
        return len(self.essay)
    
    def __getitem__(self, item):
        
        essay = str(self.essay[item])
        tokenizer_dict = self.tokenizer(
            essay,
            add_special_tokens=True,
            truncation=True,
            max_length=self.maxlen,
            padding='max_length'
        )
        ids = tokenizer_dict['input_ids']
        token_type_ids = tokenizer_dict['token_type_ids']
        attention_mask = tokenizer_dict['attention_mask']
        
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'mask': torch.tensor(attention_mask, dtype=torch.long),
        }