import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)  
import json
import re
import torch
import fsspec
import pickle
import numpy as np
import argparse
from helper.util import *
from helper.global_var import *
from helper.fid_model import *
from helper.metrics import *
from convertHTML import get_dataset
from convertHTML.utils import LexicographicSort
from collections import defaultdict
from typing import Dict
from helper.visualization import *
from tqdm import *
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.loader import DataLoader


def remove_repeat(bbox, label):
    if bbox.size(0) == 0:
        return bbox, label 
    bbox_label = torch.cat((bbox, label.unsqueeze(1)), dim=1)  
    unique_bbox_label = []  
    for item in bbox_label:   
        same_bbox_label_exists = False  
        for unique_item in unique_bbox_label:  
            if torch.all(torch.eq(item, unique_item)):  
                same_bbox_label_exists = True  
                break  
        if not same_bbox_label_exists:  
            unique_bbox_label.append(item) 
     
    unique_bbox_label = torch.stack(unique_bbox_label) 
    unique_bbox = unique_bbox_label[:, :-1]  
    unique_label = unique_bbox_label[:, -1] 
    
    return unique_bbox, unique_label


def cluster_corrd(_clustering_models, bbox, keys):
    all_res = []
    device_ = bbox.device
    if bbox.size(0) == 0:
        return bbox
    elif bbox.dim() > 2:
        for item in bbox:
            itermediate_res = []
            for i, k in enumerate(keys):
                model = _clustering_models[k]
                ins_ = item[..., i][:, None].cpu().numpy().astype(np.float32)
                cluster_i = model.predict(ins_)
                new_bbox = model.cluster_centers_[cluster_i]
                itermediate_res.append(torch.from_numpy(new_bbox).to(device_))
            all_res.append(torch.concatenate(itermediate_res, dim=1))
        all_res = torch.stack(all_res, dim=0)
    else:
        for i, k in enumerate(keys):
            model = _clustering_models[k]
            bbox_ = bbox[..., i][:, None].cpu().numpy().astype(np.float32)
            cluster_i = model.predict(bbox_)
            new_bbox = model.cluster_centers_[cluster_i]
            all_res.append(torch.from_numpy(new_bbox).to(device_))
        all_res = torch.concatenate(all_res, dim=1)
    return all_res
        

def preprocess_batch(cluster_model, layouts, max_len: int, device: torch.device, k_means=True):
    layout = defaultdict(list)
    empty_ids = []  # 0: empty 1: full
    for sample in layouts:
        if not isinstance(sample["bbox"], torch.Tensor):
            bbox, label = torch.tensor(sample["bbox"]), torch.tensor(sample["categories"])
        else:
            bbox, label = sample["bbox"], sample["categories"]
        bbox, label = remove_repeat(bbox, label)
        if k_means == True:
            bbox =  cluster_corrd(cluster_model, bbox, ['x-64', 'y-64', 'w-64', 'h-64'])
        pad_len = max_len - label.size(0)

        if pad_len == max_len:
            empty_ids.append(0)
            pad_bbox = torch.tensor(np.full((max_len, 4), 0.0), dtype=torch.float)
            pad_label = torch.tensor(np.full((max_len,), 0), dtype=torch.long)
            mask = torch.tensor([False for _ in range(max_len)])
        else:
            empty_ids.append(1)  # not empty
            pad_bbox = torch.tensor(
                np.concatenate([bbox, np.full((pad_len, 4), 0.0)], axis=0),
                dtype=torch.float,
            )
            pad_label = torch.tensor(
                np.concatenate([label, np.full((pad_len,), 0)], axis=0),
                dtype=torch.long,
            )
            mask = torch.tensor(
                [True for _ in range(bbox.shape[0])] + [False for _ in range(pad_len)]
            )

        layout["bbox"].append(pad_bbox)
        layout["label"].append(pad_label)
        layout["mask"].append(mask)
        
    bbox = torch.stack(layout["bbox"], dim=0).to(device)
    label = torch.stack(layout["label"], dim=0).to(device)
    mask = torch.stack(layout["mask"], dim=0).to(device)
    
    padding_mask = ~mask.bool()  
    return bbox, label, padding_mask, mask, empty_ids     
        

def extract_WH(s):
    pattern = r'<svg width="([\d.]+)" height="([\d.]+)">'  
    match = re.search(pattern, s)  
    W, H = match.groups()  
    return  W, H  


def extract_xywh(s):
    bboxs, labels = [], []  
    pattern = r'<rect data-category="([^"]+)".*?x="(\d+(\.\d+)?)".*?y="(\d+(\.\d+)?)".*?width="(\d+(\.\d+)?)".*?height="(\d+(\.\d+)?)".*?/>'  
    matches = re.findall(pattern, s)  
    for match in matches:  
        data_category, x, _, y, _, width, _, height, _ = match  
        label = label_to_int.get(data_category)  
        # bbox = [x, y, x + width, y + height]  
        if label is None:
            continue
        bboxs.append([eval(x), eval(y), eval(width), eval(height)])
        labels.append(label)
    return bboxs, labels


def preprocess(sample: str):
    """
    key_lst can contain:
        - cond_bbox_input
        - continual_gen_input
        - cond_cate_to_size_pos
        - cond_cate_size_to_pos
        - cond_recover_mask_input
    """
    instance = sample.split("##Here is the result:")
    if len(instance) > 2:
        res = instance[-2]
    else:
        res = instance[1]
    bboxs, categories = extract_xywh(res)
    return bboxs, categories


def convert_to_array(generations: List[str]):
    final_res = []
    for sample in generations:
        bboxs, cates = sample.get("bbox"), sample.get("categories")
        bboxs, cates = bboxs.tolist(), cates.tolist()
        bboxs = np.array(bboxs, dtype=np.float32)
        cates = np.array(cates, dtype=np.int32)
        final_res.append((bboxs, cates))
    return final_res
        
        
def print_scores(scores: Dict):
    scores = {k: scores[k] for k in sorted(scores)}

    tex = ""
    for k, v in scores.items():
        # if k == "Alignment" or k == "Overlap" or "Violation" in k:
        #     v = [_v * 100 for _v in v]
        mean, std = np.mean(v), np.std(v)
        stdp = std * 100.0 / mean
        print(f"\t{k}: {mean:.4f} ({stdp:.4f}%)")
        tex += f"& {mean:.4f}\\std{{{stdp:.1f}}}\% "
    print(tex)

def average(scores):
    return sum(scores) / len(scores)

def main(args):
    
    ## load cluster model
    _clustering_models = dict()
    valid_keys = [f"{k}-64" for k in ['x', 'y', 'w', 'h']]
    with fsspec.open(args.cluster_model, "rb") as f:
        for key, model in pickle.load(f).items():
            if key not in valid_keys:
                continue
            # sort cluster center in 1d case
            var_name = key.split("-")[0]
            if len(var_name) == 1:
                cluster_centers = np.sort(
                    model.cluster_centers_, axis=0
                )  # (N, 1)
                model.cluster_centers_ = cluster_centers
            _clustering_models[key] = model   
    
    # load val data
    print(f">>> load validation data")
    transforms = [LexicographicSort()]
    val_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='val',
            transform=T.Compose(transforms)
        )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0)

    print(">>> load testing data")
    transforms = [LexicographicSort()]
    test_dataset = get_dataset(
            name=args.dataset_name,
            datapath=args.dataset_path,
            split='test',
            transform=T.Compose(transforms)
        )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=0)
    layouts_main = loader_to_list(test_dataloader)

    # load golden data
    print(f">>> load golden file from {args.golden_file}")
    with open(args.golden_file, "r") as f:
        goldens = [json.loads(line) for line in f]
    
    # load fid model
    print(">>> load fid model")
    num_classes = len(label_to_int)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fid_model = load_fidnet_v3_simple(args.fid_model_name_or_path, device, num_label=num_classes)
    
    print(">>> Extract features from testing file")
    feats_test = []
    for i, batch in enumerate(test_dataloader):
        bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
        bbox = cluster_corrd(_clustering_models, bbox, valid_keys)
        with torch.set_grad_enabled(False):
            feat = fid_model.extract_features(bbox, label, padding_mask)
        feats_test.append(feat.cpu())
        
    print(">>> Extract features from eval file")
    feats_val = []
    for i, batch in enumerate(val_dataloader):
        bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
        with torch.set_grad_enabled(False):
            feat = fid_model.extract_features(bbox, label, padding_mask)
        feats_val.append(feat.cpu())
    
    real_fid = compute_generative_model_scores(feats_test, feats_val)
    print("real FID score is")
    print(real_fid)
    
    if args.test_others:  # implement your own evaluation code here
        pass
    
    ### check and save all generations
    if os.path.exists(args.intermediate_saved_path):  
        print(f"{args.intermediate_saved_path} already exists, read directly!")
        all_generations = torch.load(args.intermediate_saved_path)
    else:  
        print(f"{args.intermediate_saved_path} not exist, process and save the results to this path!")
        with open(args.gen_res_path, "r") as f:
            generations = [json.loads(line) for line in f]    
        all_generations = []
        for i, sample in enumerate(generations):
            if "id" in sample:
                id_ = sample.pop("id")
            else:
                id_ = i
            
            # extract corresponding golden bbox and categories
            golden_bboxs, golden_labels = extract_xywh(goldens[id_].get('labels')[0])
            W, H = extract_WH(goldens[id_].get('labels')[0])
            # construct golden features
            golden_bboxs = [convert_real_to_xywh(bbox, W, H) for bbox in golden_bboxs]
            golden_bboxs = torch.tensor(golden_bboxs, dtype=torch.float)
            golden_labels = torch.tensor(golden_labels, dtype=torch.long)
            per_sample = {
                "id": id_, 
                "golden": {
                    "bbox": golden_bboxs,
                    "categories": golden_labels,
                },
            } 
            for k, v in sample.items():
                gen_bboxs, gen_categories = preprocess(v)
                gen_bboxs = [convert_real_to_xywh(bbox, W, H) for bbox in gen_bboxs]
                gen_bboxs = torch.tensor(gen_bboxs, dtype=torch.float)
                
                gen_categories = torch.tensor(gen_categories, dtype=torch.long)
                per_sample[k] = {"bbox": gen_bboxs, "categories": gen_categories}
            all_generations.append(per_sample)
        
        torch.save(all_generations, args.intermediate_saved_path)
    
    all_keys = list(all_generations[0].keys())
    block_lst = ["id", "golden"]
    print(f">>> All keys are {all_keys} | Begin to extract features from generations")
    
    scores_all = dict()
    for gk in all_keys:
        # create saver for each k
        if gk in block_lst:
            continue
        feats_gen = [] 
        batch_metrics = defaultdict(float)
        filter_ids = [] # filter the empty bbox 
        k_generations = [tmp[gk] for tmp in all_generations]  
        for i in range(0, len(k_generations), args.batch_size):
            i_end = min(i + args.batch_size, len(k_generations))
            batch = k_generations[i:i_end]
            max_len = max(len(g["categories"]) for g in batch)
            if max_len == 0:  # prevent not generations
                max_len == 1
            bbox, label, padding_mask, mask, empty_ids = preprocess_batch(_clustering_models, batch, max_len, device, k_means=True)
            filter_ids.extend(empty_ids)
            
            mask_empty_ids = torch.tensor(empty_ids, dtype=torch.bool, device=device)
            bbox = torch.masked_select(bbox, mask_empty_ids.unsqueeze(1).unsqueeze(2)).reshape(-1, bbox.size(1), bbox.size(2)).contiguous()
            label = torch.masked_select(label, mask_empty_ids.unsqueeze(1)).reshape(-1, label.size(1)).contiguous()
            padding_mask = torch.masked_select(padding_mask, mask_empty_ids.unsqueeze(1)).reshape(-1, padding_mask.size(1)).contiguous()
            mask = torch.masked_select(mask, mask_empty_ids.unsqueeze(1)).reshape(-1, mask.size(1)).contiguous()
            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            feats_gen.append(feat.cpu())    
                
            for k, v in compute_alignment(bbox, mask).items():
                batch_metrics[k] += v.sum().item()
            for k, v in compute_overlap(bbox, mask).items():
                batch_metrics[k] += v.sum().item()  

        print(">>> Extract features from testing file again (this time filter the failed bbox)")
        filter_feats_test = []
        end_flag = False
        for i, batch in enumerate(test_dataloader):
            bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
            b_i = i * args.batch_size
            e_i = b_i + bbox.size(0)
            empty_ids = filter_ids[b_i: e_i]
            if len(empty_ids) < bbox.size(0):
                bbox = bbox[:len(empty_ids)]
                label = label[:len(empty_ids)]
                padding_mask = padding_mask[:len(empty_ids)]
                end_flag = True
            mask_empty_ids = torch.tensor(empty_ids, dtype=torch.bool, device=device)
            bbox = torch.masked_select(bbox, mask_empty_ids.unsqueeze(1).unsqueeze(2)).reshape(-1, bbox.size(1), bbox.size(2)).contiguous()
            label = torch.masked_select(label, mask_empty_ids.unsqueeze(1)).reshape(-1, label.size(1)).contiguous()
            padding_mask = torch.masked_select(padding_mask, mask_empty_ids.unsqueeze(1)).reshape(-1, padding_mask.size(1)).contiguous()
            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            filter_feats_test.append(feat.cpu())
            if end_flag:
                break
        
        # calculate IOU (max, average) and docsim score
        gen_data = convert_to_array(k_generations)
        
        scores = {}
        for k, v in batch_metrics.items():
            scores[k] = v / len(k_generations)
        
        scores.update(compute_average_iou(gen_data))
        scores.update(compute_generative_model_scores(filter_feats_test, feats_gen))
        scores["maximum_iou"] = compute_maximum_iou(layouts_main, gen_data)
        scores["DocSim"] = compute_docsim(layouts_main, gen_data)
        
        scores["Failed_rate"] = (len(filter_ids) - sum(filter_ids)) / len(filter_ids) * 100
        
        scores_all[gk] = defaultdict(list)
        for k, v in scores.items():
            scores_all[gk][k].append(v)
            
        
        print(f">>> Logging Scores for condition {gk}\n")
        print("-"*20 + "\n")
        print_scores(scores_all[gk])
     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Generated Layout Code")
    parser.add_argument("--test_others", action='store_true', help="custom your evaluation own code here")  
    parser.add_argument("--file_dir", type=str, default="data/generated_results/rico")  
    parser.add_argument("--intermediate_saved_path", type=str, default=None) 
    parser.add_argument("--golden_file", type=str, default="data/generated_results/rico/golden.jsonl")  
    parser.add_argument("--fid_model_name_or_path", type=str, default="models/rico25-max25",)  
    parser.add_argument("--cluster_model", type=str, default="models/rico25-max25/rico25_max25_kmeans_train_clusters.pkl")
    parser.add_argument("--dataset_name", type=str, default="rico25")
    parser.add_argument("--dataset_path", type=str, default="data/rico25-max25")
    parser.add_argument("--gen_res_path", type=str, default=None, help="generated html code")  
    parser.add_argument("--batch_size", type=int, default=32)  
    parser.add_argument("--device", type=str, default="cuda:0")  

    args = parser.parse_args()
    
    int_to_lable = DATASET_META.get(args.dataset_name)
    label_to_int = dict([(v, k) for k, v in int_to_lable.items()])

    main(args)