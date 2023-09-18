from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import torch
import transformers


class RawFileDataset(Dataset):  
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        with open(file, "r") as f:
            self.content = [json.loads(line) for line in f]
        self.cond_type = ["cond_cate_size", "cond_category", "cond_mask"]
        self.sample_prob = [1, 1, 1]  # custom your sampling weight here
  
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        data = self.content[index]

        cond_cate_to_size_pos_seq_modeling = data["cond_cate_to_size_pos_seq_modeling"]
        cond_cate_size_to_pos_seq_modeling = data["cond_cate_size_to_pos_seq_modeling"]
        cond_recover_mask_seq_modeling = data["cond_recover_mask_seq_modeling"]
        
        ## category -> size + pos
        cond_cate_to_size_pos_inputs = self.tokenizer(
            cond_cate_to_size_pos_seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        cond_cate_ids = cond_cate_to_size_pos_inputs.input_ids[0]
        cond_cate_attention_mask = cond_cate_to_size_pos_inputs.attention_mask[0]
        cond_cate_labels = torch.where(
            cond_cate_ids != self.tokenizer.pad_token_id, cond_cate_ids, -100
        )
        
        ## categoty + size -> pos
        cond_cate_size_inputs = self.tokenizer(
            cond_cate_size_to_pos_seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        cond_cate_size_ids = cond_cate_size_inputs.input_ids[0]
        cond_cate_size_attention_mask = cond_cate_size_inputs.attention_mask[0]
        cond_cate_size_labels = torch.where(
            cond_cate_size_ids != self.tokenizer.pad_token_id, cond_cate_size_ids, -100
        )
        
        ## random mask
        random_mask_inputs = self.tokenizer(
            cond_recover_mask_seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        random_mask_input_ids = random_mask_inputs.input_ids[0]
        random_mask_attention_mask = random_mask_inputs.attention_mask[0]
        random_mask_labels = torch.where(
            random_mask_input_ids != self.tokenizer.pad_token_id, random_mask_input_ids, -100
        )
        
        instances = {
            "cond_cate_size":{
                "input_ids": cond_cate_size_ids,
                "attention_mask": cond_cate_size_attention_mask,
                "labels":cond_cate_size_labels,
            },
            "cond_category": {
                "input_ids": cond_cate_ids,
                "attention_mask": cond_cate_attention_mask,
                "labels": cond_cate_labels,
            },
            "cond_mask": {
                "input_ids": random_mask_input_ids,
                "attention_mask": random_mask_attention_mask,
                "labels": random_mask_labels,
            }
        }
        
        ## random sampling
        selected_types = random.choices(self.cond_type, self.sample_prob, k=2)  # joint loss (t=2 here)
        instance_1 = instances[selected_types[0]]
        instance_2 = instances[selected_types[1]]

        return {
            "instance_1": instance_1,
            "instance_2": instance_2,
        }