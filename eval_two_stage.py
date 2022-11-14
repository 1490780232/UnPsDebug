import argparse
import datetime
import os.path as osp
import time
import torch
import torch.utils.data
from collections import defaultdict
from datasets import build_test_loader, build_train_loader,build_cluster_loader, build_dataset, build_transforms
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch, to_device, evaluate_performance_twostage
# from models.seqnet import SeqNet
from models.seqnet_distill import SeqNetDis as SeqNet
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler,Logger
import torch.nn.functional as F
from models.oim2 import OIMUnsupervisedLoss
import numpy as np
from sklearn.cluster import DBSCAN
from utils.compute_dist import *
import sys

def collate_fn(batch):
    return tuple(zip(*batch))

def build_cluster_loader(cfg):
    transforms = build_transforms(is_train=False)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    sys.stdout = Logger(osp.join("./", 'log_test.txt'))
    print("Loading data")
    transforms = build_transforms(is_train=True)
    gallery_loader, query_loader = build_test_loader(cfg)
    evaluate_performance_twostage(
        gallery_loader,
        query_loader,
        device,
        use_gt=cfg.EVAL_USE_GT,
        use_cache=cfg.EVAL_USE_CACHE,
        use_cbgm=cfg.EVAL_USE_CBGM,
    )
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
