import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)  
import torch
import random
import torchvision.transforms as T
import os
import json
import copy
import argparse
from convertHTML.utils import LexicographicSort
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from transformers import AutoTokenizer
from convertHTML import get_dataset
from helper.global_var import *
from collections import OrderedDict
from typing import List, Dict  
from tqdm import *
from helper.metrics import *

##################
### Global Config
##################
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
SPAN_MASK_TOKEN = "<FILL_{i}>"
SEP_TOKEN = "<sep>"
PLACE_HOLDER = "<MASK>"


def round_nested_list(nested_list, decimals):  
    result = []  
    for item in nested_list:  
        if isinstance(item, list):   
            result.append(round_nested_list(item, decimals))  
        else:  
            result.append(round(item, decimals))  
    return result 


class CustomDataLoader(DataLoader):  
    def __init__(
        self, 
        args,
        tokenizer, 
        bbox_quantization, 
        dataset, 
        batch_size,
        shuffle=False, 
        split='train',
        **kwargs
    ):  
        super(CustomDataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)  
        self.split = split
        
        self.html_template = TEMPLATE_FORMAT.get("html_format")
        self.bbox_template = TEMPLATE_FORMAT.get("bbox_format")
        
        if args.infilling:
            self.cond_cate_to_size_pos = INFILLING_INSTRUCTION.get("cond_cate_to_size_pos")
            self.cond_cate_size_to_pos = INFILLING_INSTRUCTION.get("cond_cate_size_to_pos")
            self.cond_random_mask = INFILLING_INSTRUCTION.get("cond_random_mask")
        else:
            self.cond_cate_to_size_pos = INSTRUCTION.get("cond_cate_to_size_pos")
            self.cond_cate_size_to_pos = INSTRUCTION.get("cond_cate_size_to_pos")
            self.cond_random_mask = INSTRUCTION.get("cond_random_mask")
        if args.add_task_instruction:
            task_instruction = TASK_INSTRUCTION[args.dataset_name]
            self.cond_cate_to_size_pos = task_instruction + self.cond_cate_to_size_pos
            self.cond_cate_size_to_pos = task_instruction + self.cond_cate_size_to_pos
            self.cond_random_mask = task_instruction + self.cond_random_mask
        
        self.category_map = DATASET_META[dataset.dataset_name]
        self.glue_template_train_eval = SEP_SEQ[0]
        self.glue_template_test = SEP_SEQ[1]
        self.glue_template_codegen_train = SEP_SEQ[2]
        self.glue_template_codegen_test = SEP_SEQ[3]
        
        self.tokenizer = tokenizer
        self.N_category = dataset.N_category
        self.bbox_quantization = bbox_quantization  # quanlization approachs
        self.consistency_num = args.consistency_num
        self.infilling = args.infilling
        
        
        
    def filter_invalid_num(self, lst, mask):
        new_lst = []
        for i in range(len(lst)):
            new_lst.append(lst[i][:mask[i].sum().item()])
        return new_lst
    
    
    def build_input_with_ele_dict(self, ele_dict: Dict, type=None):
        answer_notepad = []
        ele_dict = copy.deepcopy(ele_dict)
        if type == "html_content":
            ele_dict = ele_dict
        elif type == "cate_masked_html":
            answer_notepad = ele_dict["c"]
            ele_dict["c"] = PLACE_HOLDER
        elif type == "size_pos_mask_html":
            c = ele_dict["c"]
            answer_notepad = [ele_dict[k] for k in ele_dict if k != "c"]
            ele_dict = dict([(k, PLACE_HOLDER) for k in ele_dict.keys()])
            ele_dict["c"] = c
        elif type == "pos_mask_html":
            answer_notepad = [ele_dict["x"], ele_dict["y"]]
            ele_dict["x"] = PLACE_HOLDER
            ele_dict["y"] = PLACE_HOLDER
        elif type == "random_mask_html":
            random_mask_num = random.choice([3, 4]) # mask up to 80% places (categoty is not masked)
            selected_mask_element = random.sample(['x', 'y', 'w', 'h'], random_mask_num)
            answer_notepad = []
            for key in selected_mask_element:
                answer_notepad.append(ele_dict[key])
                ele_dict[key] = PLACE_HOLDER
            
        return self.bbox_template.format(**ele_dict), answer_notepad
    
    
    def replace_order_mask(self, lst: List[str], ans_lst: List):
        '''
        replace the mask token and build corresponding results
        '''
        new_lst, new_ans = [], {}
        cnt = 1
        for line, ans in zip(lst, ans_lst):
            mask_count = line.count(PLACE_HOLDER)
            for i in range(mask_count):
                mask_token = SPAN_MASK_TOKEN.format(i=cnt)
                line = line.replace(PLACE_HOLDER, mask_token, 1)
                new_ans[mask_token] = ans[i]
                cnt += 1
            new_lst.append(line)
        return new_lst, new_ans
    
    
    def convert_num_to_html(self, coord_lst=None, category_lst=None, self_consistency=False, consistency_num=10):
        batched_html_lst = []  # target
        batched_cond_cate, batched_cond_bbox = [], []  # condition
        cond_cate_to_size_pos, cond_cate_size_to_pos = [], [] 
        random_mask = []
    
        if coord_lst is not None and category_lst is not None: # create the training data   
            for coords, categories in zip(coord_lst, category_lst):
                # store all the input code
                html_content, cate_masked_html = [], []
                size_pos_mask_html, pos_mask_html = [], []
                random_mask_html = []
                
                # store all the ans
                cate_masked_html_ans, random_masked_html_ans = [], []
                size_pos_mask_html_ans, pos_mask_html_ans = [], []
                
                all_category = OrderedDict([(i, 0) for i in range(self.N_category)])
                for coord, category in zip(coords, categories):
                    w, h = coord[2], coord[3]
                    x, y = int(coord[0] - w / 2), int(coord[1] - h / 2)
                    real_category = self.category_map[category]
                    all_category[category] += 1
                    ele_dict = {"c": real_category, "x": x, "y":y, "w":w, "h":h}
                    tmp1, _ = self.build_input_with_ele_dict(ele_dict, "html_content")
                    html_content.append(tmp1)
                    
                    tmp2, ans2 = self.build_input_with_ele_dict(ele_dict, "cate_masked_html")
                    cate_masked_html.append(tmp2)
                    cate_masked_html_ans.append(ans2)
                    
                    tmp3, ans3 = self.build_input_with_ele_dict(ele_dict, "size_pos_mask_html")
                    size_pos_mask_html.append(tmp3)
                    size_pos_mask_html_ans.append(ans3)
                    
                    tmp4, ans4 = self.build_input_with_ele_dict(ele_dict, "pos_mask_html")
                    pos_mask_html.append(tmp4)
                    pos_mask_html_ans.append(ans4)
                    
                    tmp5, ans5 = self.build_input_with_ele_dict(ele_dict, "random_mask_html")
                    random_mask_html.append(tmp5)
                    random_masked_html_ans.append(ans5)
               
                ### post process the mask token id
                cate_masked_html, cate_masked_ans = self.replace_order_mask(cate_masked_html, cate_masked_html_ans)
                size_pos_mask_html, size_pos_mask_ans = self.replace_order_mask(size_pos_mask_html, size_pos_mask_html_ans)
                pos_mask_html, pos_mask_ans = self.replace_order_mask(pos_mask_html, pos_mask_html_ans)
                random_mask_html, random_mask_ans = self.replace_order_mask(random_mask_html, random_masked_html_ans)
                
                verbal_all_categories = []
                for i in range(self.N_category):
                    if all_category[i] != 0:
                        verbal_category = self.category_map[i]
                        verbal_number = VERBALIZED_NUM[all_category[i]]
                        verbal_all_categories.append("{} {},".format(verbal_number, verbal_category))
                all_verbal_all_cates = " ".join(verbal_all_categories).rstrip(",")
                
                if self_consistency == True:  # random shuffle the condition, but stay with the target
                    shuffle_lst = [i for i in range(len(html_content))]
                    min_shuffle_num = min(len(shuffle_lst), consistency_num)
                    
                    def shuffle_list(input_list):  
                        random.shuffle(input_list)  
                        return input_list  
                    
                    shuffled_results = []  
                    for i in range(min_shuffle_num):  
                        shuffled_results.append(shuffle_list(shuffle_lst.copy()))
                    
                    for random_order in shuffled_results:
                        # new_html_content = [html_content[i] for i in random_order]
                        new_cate_masked_html = [cate_masked_html[i] for i in random_order]
                        new_size_pos_mask_html = [size_pos_mask_html[i] for i in random_order]
                        new_pos_mask_html = [pos_mask_html[i] for i in random_order]
                        new_random_mask_html = [random_mask_html[i] for i in random_order]

                        batched_cond_cate.append(all_verbal_all_cates)   
                        batched_html_lst.append("\n".join(html_content))  # save target
                        batched_cond_bbox.append('\n'.join(new_cate_masked_html))
                        cond_cate_to_size_pos.append("\n".join(new_size_pos_mask_html))
                        cond_cate_size_to_pos.append("\n".join(new_pos_mask_html))
                        random_mask.append("\n".join(new_random_mask_html))  
                        
                else:
                    # process all conditions
                    batched_cond_cate.append(all_verbal_all_cates)   
                    batched_cond_bbox.append('\n'.join(cate_masked_html))
                    batched_html_lst.append("\n".join(html_content))
                    cond_cate_to_size_pos.append("\n".join(size_pos_mask_html))
                    cond_cate_size_to_pos.append("\n".join(pos_mask_html))
                    random_mask.append("\n".join(random_mask_html))
                
        else:
            raise ValueError("Can not inplement to testing data")

        return {
            "batched_html_lst": batched_html_lst,
            "batched_cond_cate": batched_cond_cate,
            "batched_cond_bbox": batched_cond_bbox,
            "cond_cate_to_size_pos": cond_cate_to_size_pos,
            "cond_cate_size_to_pos": cond_cate_size_to_pos,
            "random_mask": random_mask,
            "codegen_ans": {
                "cate_masked_ans": cate_masked_ans,
                "size_pos_mask_ans": size_pos_mask_ans,
                "pos_mask_ans": pos_mask_ans,
                "random_mask_ans": random_mask_ans,
            },
        }
    
    
    def build_random_mask(self, lst):
        new_lst = lst.copy()
        num = random.sample([3, 4], 1)[0]  # mask up to 80% position
        pos = random.sample([0,1,2,3], num)
        for i in pos:
            new_lst[i] = PLACE_HOLDER
        return new_lst
    
    
    def generate_new_order(self, lst):
        shuffle_order = [i for i in range(len(lst))]
        random.shuffle(shuffle_order)
        return shuffle_order
        
                
    def custom_function(self, data, id_, self_consistency=True, consistency_num=10):  
        label, mask = to_dense_batch(data.y, data.batch)   # (B, S)
        bbox_real, _ = to_dense_batch(data.x, data.batch)  # (B, S, 4)
        W, H = data.attr["width"], data.attr["height"]
        
        size_ = torch.cat((W.unsqueeze(-1), H.unsqueeze(-1), W.unsqueeze(-1), H.unsqueeze(-1)), dim=-1)
        size_ = size_.unsqueeze(1)
        real_idx = size_ * bbox_real # [cx, cy, w, h]
        if self.bbox_quantization == "code":
            label = label.to(torch.int).tolist()
            label_lst = self.filter_invalid_num(label, mask)
            real_idx = real_idx.to(torch.float).tolist()
            
            real_idx = round_nested_list(real_idx, 1)
            bbox_lst = self.filter_invalid_num(real_idx, mask)
            preposed_res = self.convert_num_to_html(
                bbox_lst, label_lst, self_consistency=self_consistency, consistency_num=consistency_num
            )
            batched_html_lst = preposed_res.get("batched_html_lst")
            batched_cond_cate = preposed_res.get("batched_cond_cate")
            batched_cond_bbox = preposed_res.get("batched_cond_bbox")   
            
            cond_cate_to_size_pos = preposed_res.get("cond_cate_to_size_pos")
            cond_cate_to_size_pos_res_dict = preposed_res["codegen_ans"].get("size_pos_mask_ans")
            
            cond_cate_size_to_pos = preposed_res.get("cond_cate_size_to_pos")
            cond_cate_size_to_pos_res_dict = preposed_res["codegen_ans"].get("pos_mask_ans")
            
            random_mask = preposed_res.get("random_mask")
            random_mask_res_dict = preposed_res["codegen_ans"].get("random_mask_ans")
        
        elif self.bbox_quantization == "numerical":
            label = label.to(torch.int).tolist()
            label_lst = self.filter_invalid_num(label, mask)[0]
            real_idx = real_idx.to(torch.float).tolist()
            real_idx = round_nested_list(real_idx, 1)
            bbox_lst = self.filter_invalid_num(real_idx, mask)[0]
            label_lst = [str(i) for i in label_lst]
            bbox_lst = [[str(i) for i in item] for item in bbox_lst]
            
            if self.split == "train" or self.split == "val":
                consistency_num = min(len(label_lst), consistency_num)
            else:
                consistency_num = 1

            final_res = []
            for j in range(consistency_num):
                if j != 0:
                    shuffle_order = self.generate_new_order(label_lst)
                    new_bbox_lst = [bbox_lst[i] for i in shuffle_order]
                    new_label_lst = [label_lst[i] for i in shuffle_order]
                else:
                    new_bbox_lst = bbox_lst
                    new_label_lst = label_lst
            
                golden_layouts = [[new_label_lst[i]] + bbox_lst[i] for i in range(len(new_label_lst))]
                golden_layouts = " ".join([" ".join(item) for item in golden_layouts])
                
                # buld size and position mask
                cond_cate_to_size_pos_masks = [[PLACE_HOLDER] * 4 for i in range(len(new_label_lst))]
                cond_cate_to_size_pos_modeling = [[new_label_lst[i]] + cond_cate_to_size_pos_masks[i] for i in range(len(new_label_lst))]
                cond_cate_to_size_pos_modeling = [" ".join(item) for item in cond_cate_to_size_pos_modeling]
                cond_cate_to_size_pos_modeling = " ".join(cond_cate_to_size_pos_modeling)
                
                # build position mask
                cond_cate_to_size_pos_masks = [[PLACE_HOLDER] * 2 + new_bbox_lst[i][2:] for i in range(len(new_label_lst))]
                cond_cate_to_size_pos_masks_modeling = [[new_label_lst[i]] + cond_cate_to_size_pos_masks[i] for i in range(len(new_label_lst))]
                cond_cate_to_size_pos_masks_modeling = [" ".join(item) for item in cond_cate_to_size_pos_masks_modeling]
                cond_cate_to_size_pos_masks_modeling = " ".join(cond_cate_to_size_pos_masks_modeling)
                
                # build random mask
                cond_recover_masks = [self.build_random_mask(item) for item in new_bbox_lst]
                cond_recover_masks_modeling = [[new_label_lst[i]] + cond_recover_masks[i] for i in range(len(new_label_lst))]
                cond_recover_masks_modeling = [" ".join(item) for item in cond_recover_masks_modeling]
                cond_recover_masks_modeling = " ".join(cond_recover_masks_modeling)
                
                if self.split == "train" or self.split == "val":
                    cond_cate_to_size_pos_modeling = cond_cate_to_size_pos_modeling + " <MID> " + golden_layouts
                    cond_cate_to_size_pos_masks_modeling = cond_cate_to_size_pos_masks_modeling + " <MID> " + golden_layouts
                    cond_recover_masks_modeling = cond_recover_masks_modeling + " <MID> " + golden_layouts
                    final_res.append({
                        "cond_cate_to_size_pos_seq_modeling": cond_cate_to_size_pos_modeling,
                        "cond_cate_size_to_pos_seq_modeling": cond_cate_to_size_pos_masks_modeling,
                        "cond_recover_mask_seq_modeling": cond_recover_masks_modeling,
                    })
                else:
                    return {
                        "cond_cate_to_size_pos_seq_modeling": cond_cate_to_size_pos_modeling + " <MID>",
                        "cond_cate_size_to_pos_seq_modeling": cond_cate_to_size_pos_masks_modeling + " <MID>",
                        "cond_recover_mask_seq_modeling": cond_recover_masks_modeling + " <MID>",
                    }

            return final_res
            
            
        if self_consistency:  # resize W and H
            W = W.repeat(len(batched_html_lst))
            H = H.repeat(len(batched_html_lst))
        
        # construct the html input 
        batched_cond_bbox = [
            self.html_template.format(W=W[i], H=H[i], content=batched_cond_bbox[i])
            for i in range(len(batched_cond_bbox))                 
        ]
        cond_cate_to_size_pos = [
            self.html_template.format(W=W[i], H=H[i], content=cond_cate_to_size_pos[i])
            for i in range(len(cond_cate_to_size_pos))  
        ]
        cond_cate_size_to_pos = [
            self.html_template.format(W=W[i], H=H[i], content=cond_cate_size_to_pos[i])
            for i in range(len(cond_cate_size_to_pos))  
        ]
        cond_recover_mask = [
            self.html_template.format(W=W[i], H=H[i], content=random_mask[i])
            for i in range(len(random_mask))  
        ]
        
        # add task instructions
        cond_recover_mask = [
            self.cond_random_mask.format(bbox_html=bbox)
            for bbox in cond_recover_mask
        ]
        cond_cate_to_size_pos = [
            self.cond_cate_to_size_pos.format(bbox_html=bbox)
            for bbox in cond_cate_to_size_pos
        ]
        cond_cate_size_to_pos = [
            self.cond_cate_size_to_pos.format(bbox_html=bbox)
            for bbox in cond_cate_size_to_pos
        ]
        bbox_cond_seqs = [
            self.cond_bbox_prefix.format(categories=cate, bbox_html=bbox_html) 
            for cate, bbox_html in zip(batched_cond_cate, batched_cond_bbox)
        ]
        category_cond_seqs = [
            self.cond_cate_prefix.format(categories=batched_cond_cate[i], W=W[i], H=H[i]) 
            for i in range(len(batched_cond_cate))
        ]

        if self.infilling and self.split in ("train", "val"):  # do infilling task
            cond_cate_to_size_pos_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in cond_cate_to_size_pos_res_dict.items())]
            cond_cate_size_to_pos_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in cond_cate_size_to_pos_res_dict.items())]
            random_mask_res_dict_golden = [f" {SEP_TOKEN} ".join(f"{key} {value}" for key, value in random_mask_res_dict.items())]
        
        # build target seq
        if self.split == "train" or self.split == "val":
            if self.infilling:
                if self_consistency:
                    consistency_num = len(cond_cate_to_size_pos)
                    target_seqs = [
                        cond_cate_to_size_pos_golden * consistency_num, 
                        cond_cate_size_to_pos_golden * consistency_num, 
                        random_mask_res_dict_golden * consistency_num
                    ]
                else:
                    target_seqs = [cond_cate_to_size_pos_golden, cond_cate_size_to_pos_golden, random_mask_res_dict_golden]
                
                cond_cate_to_size_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_to_size_pos, target_seqs[0])
                ]
                
                cond_cate_size_to_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_size_to_pos, target_seqs[1])
                ]
                
                cond_recover_mask_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_recover_mask, target_seqs[2])
                ]

                return {
                    "cond_cate_to_size_pos_seq_modeling": cond_cate_to_size_pos_seq_modeling,
                    "cond_cate_size_to_pos_seq_modeling": cond_cate_size_to_pos_seq_modeling,
                    "cond_recover_mask_seq_modeling": cond_recover_mask_seq_modeling,
                }
                
            else:
                target_seqs = [
                    self.html_template.format(W=W[i], H=H[i], content=batched_html_lst[i])
                    for i in range(W.size(0))
                ]
            
                cond_recover_mask_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_recover_mask, target_seqs)
                ]
                
                cond_cate_to_size_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_to_size_pos, target_seqs)
                ]
                
                cond_cate_size_to_pos_seq_modeling = [
                    self.glue_template_codegen_train.format(instruct=sample, result=target)
                    for sample, target in zip(cond_cate_size_to_pos, target_seqs)
                ]
            
                return {
                    "cond_cate_to_size_pos_seq_modeling": cond_cate_to_size_pos_seq_modeling,
                    "cond_cate_size_to_pos_seq_modeling": cond_cate_size_to_pos_seq_modeling,
                    "cond_recover_mask_seq_modeling": cond_recover_mask_seq_modeling,
                }
            
        else:
            if self.infilling:
                cond_bbox_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in bbox_cond_seqs
                ]
                
                continual_gen_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in category_cond_seqs
                ]
                
                cond_cate_size_to_pos_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_size_to_pos
                ]
                
                cond_cate_to_size_pos_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_cate_to_size_pos
                ]
                
                cond_recover_mask_input_seqs = [
                    self.glue_template_codegen_test.format(instruct=instance)
                    for instance in cond_recover_mask
                ]
            
            else:
                cond_bbox_input_seqs = [
                    self.glue_template_test.format(instruct=instance)
                    for instance in bbox_cond_seqs
                ]
                
                continual_gen_input_seqs = [
                    self.glue_template_test.format(instruct=instance)
                    for instance in category_cond_seqs
                ]
                
                cond_cate_size_to_pos_input_seqs = [
                    self.glue_template_test.format(instruct=instance)
                    for instance in cond_cate_size_to_pos
                ]
                
                cond_cate_to_size_pos_input_seqs = [
                    self.glue_template_test.format(instruct=instance)
                    for instance in cond_cate_to_size_pos
                ]
                
                cond_recover_mask_input_seqs = [
                    self.glue_template_test.format(instruct=instance)
                    for instance in cond_recover_mask
                ]
            
            
            labels = None
            if batched_html_lst is not None:
                
                labels = [
                    self.html_template.format(W=W[i], H=H[i], content=batched_html_lst[i])
                    for i in range(W.size(0))
                ]
            
            return {
                "cond_bbox_input_seqs": cond_bbox_input_seqs,
                "continual_gen_input_seqs": continual_gen_input_seqs,
                "cond_cate_size_to_pos_input_seqs": cond_cate_size_to_pos_input_seqs,
                "cond_cate_to_size_pos_input_seqs": cond_cate_to_size_pos_input_seqs,
                "cond_recover_mask_input_seqs": cond_recover_mask_input_seqs,
                "labels": labels,
                "raw_data": {
                    "category": bbox_lst[0],
                    "bbox": label_lst[0]
                },
                "id_": id_
            }
    
    def __iter__(self):
        for i, data in enumerate(super(CustomDataLoader, self).__iter__()):  
            if self.consistency_num > 1:
                self_consistency = True
            else:
                self_consistency = False
            yield self.custom_function(data, i, self_consistency=self_consistency)  

    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id
    
    @property
    def pad_token_id(self) -> int: 
        return self.tokenizer.pad_token_id
    
    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.unk_token_id
    
    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Layout Code in HTML format")
    parser.add_argument("--dataset_name", type=str, default="rico25", help="dataset name")  
    parser.add_argument("--dataset_path", type=str, default="data/rico25-max25")
    parser.add_argument("--save_path", type=str, default="data/rico25-max25/html_format")
    parser.add_argument("--model_path_or_name", type=str, default=None, help="tokenizer model name")
    parser.add_argument("--bbox_quantization", type=str, default="code", choices=["code", "numerical"])
    parser.add_argument("--consistency_num", type=int, default=1, help="number of consistency num")
    parser.add_argument("--build_testing_set", action="store_true", help="whether to build the testing set")
    parser.add_argument("--infilling", action="store_true", help="whether to build the infilling data set")
    parser.add_argument("--add_task_instruction", action="store_true", help="whether to add the task instruction")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):  
        os.makedirs(args.save_path)  
        print(f"Directory '{args.save_path}' created.")  
    
    if args.model_path_or_name is None:
        print("please specify the model name or path")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
    transforms = [LexicographicSort()]
     
    if not args.build_testing_set:
        train_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='train',
            transform=T.Compose(transforms)
        )
        
        eval_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='val',
            transform=T.Compose(transforms)
        )
        
        train_dataloader = CustomDataLoader(
            args,
            tokenizer,
            bbox_quantization=args.bbox_quantization,
            dataset=train_dataset, 
            batch_size=1,
            split="train",
        )
        
        eval_dataloader = CustomDataLoader(
            args,
            tokenizer,
            bbox_quantization=args.bbox_quantization,
            dataset=eval_dataset, 
            batch_size=1,
            split="val",
        )
        
        all_train_data, all_eval_data = [], []
    
        train_file = os.path.join(args.save_path, "train_llama_numerical.jsonl")
        val_file = os.path.join(args.save_path, "val_llama_numerical.jsonl")
        
        print(f"begin to save train file >>> {args.save_path}")
        with tqdm(total=len(train_dataloader)) as pbar:
            for i, batch_inputs in enumerate(train_dataloader):
                if args.consistency_num > 1:
                    inner_batch = len(batch_inputs['cond_cate_to_size_pos_seq_modeling'])
                    new_batch_inputs = [{} for i in range(inner_batch)]
                    for k, v in batch_inputs.items():
                        for i, value in enumerate(v):
                            new_batch_inputs[i][k] = value    
                    batch_inputs = new_batch_inputs
                else:
                    batch_inputs = [batch_inputs]
                all_train_data.extend(batch_inputs)
                pbar.update(1)
        with open(train_file, "w") as f:
            for line in all_train_data:
                f.write(json.dumps(line) + "\n")
                
        print(f"training data saved done, begin to save eval dataset >>> {args.save_path}")
        with tqdm(total=len(eval_dataloader)) as pbar:
            for i, batch_inputs in enumerate(eval_dataloader):
                if args.consistency_num > 1:
                    inner_batch = len(batch_inputs['cond_cate_to_size_pos_seq_modeling'])
                    new_batch_inputs = [{} for i in range(inner_batch)]
                    for k, v in batch_inputs.items():
                        for i, value in enumerate(v):
                            new_batch_inputs[i][k] = value    
                    batch_inputs = new_batch_inputs
                else:
                    batch_inputs = [batch_inputs]
                all_eval_data.extend(batch_inputs)
                pbar.update(1)
        with open(val_file, "w") as f:
            for line in all_eval_data:
                f.write(json.dumps(line) + "\n")
    
    else:
        test_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='test',
            transform=T.Compose(transforms)
        )
        
        test_dataloader = CustomDataLoader(
            args,
            tokenizer,
            bbox_quantization=args.bbox_quantization,
            dataset=test_dataset, 
            batch_size=1,
            split="test",
        )
    
        all_test_data = []
        test_file = os.path.join(args.save_path, "test_numerical.jsonl")
        print("begin to save test file")
        with tqdm(total=len(test_dataloader)) as pbar:
            for i, batch_inputs in enumerate(test_dataloader):
                all_test_data.append(batch_inputs)
                pbar.update(1)
        with open(test_file, "w") as f:
            for line in all_test_data:
                f.write(json.dumps(line) + "\n")
    
    
    