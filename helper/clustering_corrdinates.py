import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)  

import argparse
import logging
import pickle
import time
import torch
from sklearn.cluster import KMeans
from convertHTML.cluster import Percentile
from convertHTML import get_dataset

logger = logging.getLogger(__name__)

KEYS = ["x", "y", "w", "h"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='publaynet')
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--max_seq_length", type=int, default=25)  
parser.add_argument("--algorithm", '-a', type=str, choices=["kmeans", "percentile"])
parser.add_argument("--result_dir", '-rd', type=str, default=None)
parser.add_argument("--random_state", type=int, default=0)
parser.add_argument(
    "--max_bbox_num",
    type=int,
    default=int(1e5),
    help="filter number of bboxes to avoid too much time consumption in kmeans",
)

args = parser.parse_args()

n_clusters_list = [2**i for i in range(1, 9)]

dataset = get_dataset(args.dataset_name, args.dataset_dir, split="train", transform=None)
bboxes = torch.cat([e.x for e in dataset], axis=0)

models = {}
# name = Path(args.dataset_yaml).stem
weight_path = f"{args.result_dir}/{args.dataset_name}_max{args.max_seq_length}_{args.algorithm}_train_clusters.pkl"

if bboxes.size(0) > args.max_bbox_num and args.algorithm == "kmeans":
    text = f"{bboxes.size(0)} -> {args.max_bbox_num}"
    logger.warning(
        f"Subsampling bboxes because there are too many for kmeans: ({text})"
    )
    generator = torch.Generator().manual_seed(args.random_state)
    indices = torch.randperm(bboxes.size(0), generator=generator)
    bboxes = bboxes[indices[: args.max_bbox_num]]

for n_clusters in n_clusters_list:
    start_time = time.time()
    if args.algorithm == "kmeans":
        kwargs = {"n_clusters": n_clusters, "random_state": args.random_state}
        # one variable
        for i, key in enumerate(KEYS):
            key = f"{key}-{n_clusters}"
            models[key] = KMeans(**kwargs).fit(bboxes[..., i : i + 1].numpy())
    elif args.algorithm == "percentile":
        kwargs = {"n_clusters": n_clusters}
        for i, key in enumerate(KEYS):
            key = f"{key}-{n_clusters}"
            models[key] = Percentile(**kwargs).fit(bboxes[..., i : i + 1].numpy())
    print(
        f"{args.dataset_name} ({args.algorithm} {n_clusters} clusters): {time.time() - start_time}s"
    )

with open(weight_path, "wb") as f:
    pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)