import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory) 

import argparse
from pycocotools.coco import COCO
from collections import defaultdict 
from tqdm import trange


def cal_box_bins(coco_instance):
      
    annotations = coco_instance.getAnnIds()
    image_layout_count = defaultdict(int)  
  
    for i in trange(len(annotations)):  
        ann_id = annotations[i]
        ann = coco_instance.loadAnns(ann_id)[0]  
        image_layout_count[ann["image_id"]] += 1  
  
    max_layouts_image_id = max(image_layout_count, key=image_layout_count.get)  
    max_layouts_count = image_layout_count[max_layouts_image_id] 
    
    return max_layouts_image_id, max_layouts_count 
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to statistic the layout dataset")  
  
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the input file")  
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Path to the output file")  
    parser.add_argument("--num_lines", "-n", type=int, default=10, help="Number of lines to extract from the input file (default: 10)")  
  
    args = parser.parse_args()  
    coco_instance = COCO(args.input_file)

    id_, max_bbox_bins = cal_box_bins(coco_instance)
    print(f">>> max bbox bins {max_bbox_bins} | id is: {id_}")
    
    
    
    

