import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)  
import copy
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from torch_geometric.data import Data
from convertHTML.utils import sparse_to_dense
from convertHTML.utils import convert_xywh_to_ltrb

from convertHTML.base import BaseDataset

magazine_labels = [
    "text",
    "image",
    "headline",
    "text-over-image",
    "headline-over-image",
]

label2index = {
    "text": 0,
    "image": 1,
    "headline": 2,
    "text-over-image": 3,
    "headline-over-image": 4,
}



def append_child(element, elements):
    if "children" in element.keys():
        for child in element["children"]:
            elements.append(child)
            elements = append_child(child, elements)
    return elements


class MagezineDataset(BaseDataset):
    def __init__(self, datapath, split='train', transform=None, max_bbox_bins=32):
        super().__init__(datapath, split, transform)
        self.N_category = 5
        self.max_box_bins = max_bbox_bins
        self.dataset_name = "magazine"

    def process(self):
        for split_publaynet in ['train', 'val']:
            data_list = []
            with open(os.path.join(self.path, f'{split_publaynet}.json'), "r") as f:
                content = json.load(f)
            for k, v in content.items():
                W = 225.0
                H = 300.0
                name = k
                
                def is_valid(element):
                    x1, y1, width, height = element['bbox']
                    x2, y2 = x1 + width, y1 + height
                    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                        return False

                    if x2 <= x1 or y2 <= y1:
                        return False

                    return True
                
                N = len(v)
                if N == 0 or 32 < N:
                    continue
                
                boxes, labels = [], []
                
                for ele_name, ele_val in v.items():
                    # bbox
                    for ele in ele_val:
                        if len(ele) > 4:
                            import pdb; pdb.set_trace()
                        x1, y1 = ele[0][0], ele[0][1]
                        width = abs(ele[2][0] - ele[0][0]) 
                        height = abs(ele[2][1] - ele[0][1])
                        xc = x1 + width / 2.
                        yc = y1 + height / 2.
                        b = [xc / W, yc / H, width / W, height / H]
                        boxes.append(b)
                        labels.append(label2index[ele_name])

                boxes = torch.tensor(boxes, dtype=torch.float)  # xc, yc, W, H
                labels = torch.tensor(labels, dtype=torch.long)

                data = Data(x=boxes, y=labels)
                data.attr = {
                    'name': name,
                    'width': W,
                    'height': H,
                    'filtered': None,
                    'has_canvas_element': False,
                    "NoiseAdded": False,
                }
                data_list.append(data)
            
            s_t_v = int(len(data_list) * .95)
            print(s_t_v)
            if split_publaynet == 'train':
                train_list = data_list[:s_t_v]
            else:
                val_list = data_list[s_t_v:]

        # shuffle train with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(train_list), generator=generator)
        train_list = [train_list[i] for i in indices]
        
        # train_list -> train 95% / val 5%
        # val_list -> test 100%
        s = int(len(train_list) * .95)
        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(train_list[s:]), self.processed_paths[1])
        torch.save(self.collate(val_list), self.processed_paths[2])


    def get_original_resource(self, batch) -> Image:
        assert not self.raw_dir.startswith("gs://")
        bbox, _, _, _ = sparse_to_dense(batch)

        img_bg, img_original, cropped_patches = [], [], []
        names = batch.attr["name"]
        if isinstance(names, str):
            names = [names]

        for i, name in enumerate(names):
            name = Path(name).name.replace(".json", ".jpg")
            img = Image.open(Path(self.raw_dir) / "combined" / name)
            img_original.append(copy.deepcopy(img))

            W, H = img.size
            ltrb = convert_xywh_to_ltrb(bbox[i].T.numpy())
            left, right = (ltrb[0] * W).astype(np.uint32), (ltrb[2] * W).astype(
                np.uint32
            )
            top, bottom = (ltrb[1] * H).astype(np.uint32), (ltrb[3] * H).astype(
                np.uint32
            )
            draw = ImageDraw.Draw(img)
            patches = []
            for (l, r, t, b) in zip(left, right, top, bottom):
                patches.append(img.crop((l, t, r, b)))
                draw.rectangle([(l, t), (r, b)], fill=(255, 255, 255))
            img_bg.append(img)
            cropped_patches.append(patches)


        return {
            "img_bg": img_bg,
            "img_original": img_original,
            "cropped_patches": cropped_patches,
        }
