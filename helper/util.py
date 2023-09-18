import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

import torch
import torchvision.utils as vutils
import torchvision.transforms as T
from torch_geometric.utils import to_dense_batch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed:", seed)


def init_experiment(args, prefix):
    if args.seed is None:
        args.seed = random.randint(0, 10000)

    set_seed(args.seed)

    if not args.name:
        args.name = datetime.now().strftime('%Y%m%d%H%M%S%f')

    out_dir = Path('output') / args.dataset / prefix / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'args.json'
    with json_path.open('w') as f:
        json.dump(vars(args), f, indent=2)

    return out_dir


def save_checkpoint(state, is_best, out_dir):
    out_path = Path(out_dir) / 'checkpoint.pth.tar'
    torch.save(state, out_path)

    if is_best:
        best_path = Path(out_dir) / 'model_best.pth.tar'
        shutil.copyfile(out_path, best_path)


def loader_to_list(loader: torch.utils.data.dataloader.DataLoader):
    layouts = []
    for batch in loader:
        bbox, label, _, mask = sparse_to_dense(batch)
        for i in range(len(label)):
            valid = mask[i].numpy()
            layouts.append((bbox[i].numpy()[valid], label[i].numpy()[valid]))
    return layouts

def sparse_to_dense(
    batch,
    device: torch.device = torch.device("cpu"),
    remove_canvas: bool = False,
):
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


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def convert_real_to_xywh(bbox, W, H):
    bbox = [float(i) for i in bbox]
    W = eval(W)
    H = eval(H)
    x1, y1, width, height = bbox
    
    xc = x1 + width / 2.
    yc = y1 + height / 2.
    b = [xc / W, yc / H, width / W, height / H]
    return b


def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color,
                       fill=c_fill)
    return img


def save_image(batch_boxes, batch_labels, batch_mask,
               dataset_colors, out_path, canvas_size=(60, 40),
               nrow=None):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes = batch_boxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        img = convert_layout_to_image(boxes, labels,
                                      dataset_colors,
                                      canvas_size)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    vutils.save_image(image, out_path, normalize=False, nrow=nrow)