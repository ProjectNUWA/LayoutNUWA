from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch


class RawFileDataset(Dataset):  
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.max_seq_length
        with open(file, "r") as f:
            self.content = [json.loads(line) for line in f]
            
  
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        data = self.content[index]
        bbox_seq_modeling = data["bbox_seq_modeling"][0]
        cate_seq_modeling = data["cate_seq_modeling"][0]
        bbox_inputs = self.tokenizer(
            bbox_seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        bbox_input_ids = bbox_inputs.input_ids[0]
        bbox_attention_mask = bbox_inputs.attention_mask[0]
        bbox_labels = torch.where(bbox_input_ids != self.tokenizer.pad_token_id, bbox_input_ids, -100)
        
        cate_inputs = self.tokenizer(
            cate_seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        cate_input_ids = cate_inputs.input_ids[0]
        cate_attention_mask = cate_inputs.attention_mask[0]
        cate_labels = torch.where(cate_input_ids != self.tokenizer.pad_token_id, cate_input_ids, -100)
        
        if random.random() < 0.5:
            return {
                "input_ids": bbox_input_ids,
                "attention_mask": bbox_attention_mask,
                "labels": bbox_labels,
            }
        else:
            return {
                "input_ids": cate_input_ids,
                "attention_mask": cate_attention_mask,
                "labels": cate_labels,
            }
    
    
    
    