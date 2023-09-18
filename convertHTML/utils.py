import torch
import random
from enum import IntEnum
from itertools import product, combinations
from torch_geometric.utils import to_dense_batch
from typing import Tuple, List
from torch import FloatTensor, LongTensor, BoolTensor, BoolTensor
import numpy as np


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


class RelSize(IntEnum):
    UNKNOWN = 0
    SMALLER = 1
    EQUAL = 2
    LARGER = 3


class RelLoc(IntEnum):
    UNKNOWN = 4
    LEFT = 5
    TOP = 6
    RIGHT = 7
    BOTTOM = 8
    CENTER = 9

REL_SIZE_ALPHA = 0.1


def detect_size_relation(b1, b2):
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    a1_sm = (1 - REL_SIZE_ALPHA) * a1
    a1_lg = (1 + REL_SIZE_ALPHA) * a1

    if a2 <= a1_sm:
        return RelSize.SMALLER

    if a1_sm < a2 and a2 < a1_lg:
        return RelSize.EQUAL

    if a1_lg <= a2:
        return RelSize.LARGER

    raise RuntimeError(b1, b2)


def detect_loc_relation(b1, b2, canvas=False):
    if canvas:
        yc = b2[1]
        y_sm, y_lg = 1. / 3, 2. / 3

        if yc <= y_sm:
            return RelLoc.TOP

        if y_sm < yc and yc < y_lg:
            return RelLoc.CENTER

        if y_lg <= yc:
            return RelLoc.BOTTOM

    else:
        l1, t1, r1, b1 = convert_xywh_to_ltrb(b1)
        l2, t2, r2, b2 = convert_xywh_to_ltrb(b2)

        if b2 <= t1:
            return RelLoc.TOP

        if b1 <= t2:
            return RelLoc.BOTTOM

        if t1 < b2 and t2 < b1:
            if r2 <= l1:
                return RelLoc.LEFT

            if r1 <= l2:
                return RelLoc.RIGHT

            if l1 < r2 and l2 < r1:
                return RelLoc.CENTER

    raise RuntimeError(b1, b2, canvas)


def get_rel_text(rel, canvas=False):
    if type(rel) == RelSize:
        index = rel - RelSize.UNKNOWN - 1
        if canvas:
            return [
                'within canvas',
                'spread over canvas',
                'out of canvas',
            ][index]

        else:
            return [
                'larger than',
                'equal to',
                'smaller than',
            ][index]

    else:
        index = rel - RelLoc.UNKNOWN - 1
        if canvas:
            return [
                '', 'at top',
                '', 'at bottom',
                'at middle',
            ][index]

        else:
            return [
                'right to', 'below',
                'left to', 'above',
                'around',
            ][index]


class LexicographicSort():
    def __call__(self, data):
        assert not data.attr['has_canvas_element']
        l, t, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data

class AddNoiseToBBox:
    def __init__(self, std: float = 0.05):
        self.std = float(std)

    def __call__(self, data):
        noise = torch.normal(0, self.std, size=data.x.size(), device=data.x.device)
        data.x_orig = data.x.clone()
        # print(f"add a noise >>>>>> {noise}\n original data is {data.x}")
        data.x = data.x + noise
        # prevent data.x smaller than 0 or greater than one
        data.x = data.x.clamp(0, 1)
        data.attr = data.attr.copy()
        data.attr["NoiseAdded"][0] = True
        return data
    
class LexicographicOrder:
    def __call__(self, data):
        assert not data.attr["has_canvas_element"]
        x, y, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(y, x)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data    
    

class HorizontalFlip():
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 0] = 1 - data.x[:, 0]
        return data


class AddCanvasElement():
    def __init__(self):
        self.x = torch.tensor([[.5, .5, 1., 1.]], dtype=torch.float)
        self.y = torch.tensor([0], dtype=torch.long)

    def __call__(self, data):
        if not data.attr['has_canvas_element']:
            data.x = torch.cat([self.x, data.x], dim=0)
            data.y = torch.cat([self.y, data.y + 1], dim=0)
            data.attr = data.attr.copy()
            data.attr['has_canvas_element'] = True
        return data


class AddRelation():
    def __init__(self, seed=None, ratio=0.1):
        self.ratio = ratio
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)

    def __call__(self, data):
        N = data.x.size(0)
        has_canvas = data.attr['has_canvas_element']

        rel_all = list(product(range(2), combinations(range(N), 2)))
        size = int(len(rel_all) * self.ratio)
        rel_sample = set(self.generator.sample(rel_all, size))

        edge_index, edge_attr = [], []
        rel_unk = 1 << RelSize.UNKNOWN | 1 << RelLoc.UNKNOWN
        for i, j in combinations(range(N), 2):
            bi, bj = data.x[i], data.x[j]
            canvas = data.y[i] == 0 and has_canvas

            if (0, (i, j)) in rel_sample:
                rel_size = 1 << detect_size_relation(bi, bj)
            else:
                rel_size = 1 << RelSize.UNKNOWN

            if (1, (i, j)) in rel_sample:
                rel_loc = 1 << detect_loc_relation(bi, bj, canvas)
            else:
                rel_loc = 1 << RelLoc.UNKNOWN

            rel = rel_size | rel_loc
            if rel != rel_unk:
                edge_index.append((i, j))
                edge_attr.append(rel)

        data.edge_index = torch.as_tensor(edge_index).long()
        data.edge_index = data.edge_index.t().contiguous()
        data.edge_attr = torch.as_tensor(edge_attr).long()

        return data


def sparse_to_dense(
    batch,
    device: torch.device = torch.device("cpu"),
    remove_canvas: bool = False,
) -> Tuple[FloatTensor, LongTensor, BoolTensor, BoolTensor]:
    batch = batch.to(device)
    bbox, _ = to_dense_batch(batch.x, batch.batch)
    label, mask = to_dense_batch(batch.y, batch.batch)

    if remove_canvas:
        bbox = bbox[:, 1:].contiguous()
        label = label[:, 1:].contiguous() - 1  # cancel +1 effect in transform
        label = label.clamp(min=0)
        mask = mask[:, 1:].contiguous()

    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def loader_to_list(
    loader: torch.utils.data.dataloader.DataLoader,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    layouts = []
    for batch in loader:
        bbox, label, _, mask = sparse_to_dense(batch)
        for i in range(len(label)):
            valid = mask[i].numpy()
            layouts.append((bbox[i].numpy()[valid], label[i].numpy()[valid]))
    return layouts


def split_num_samples(N: int, batch_size: int) -> List[int]:
    quontinent = N // batch_size
    remainder = N % batch_size
    dataloader = quontinent * [batch_size]
    if remainder > 0:
        dataloader.append(remainder)
    return dataloader