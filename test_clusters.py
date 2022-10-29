import argparse
import datetime
import os.path as osp
import time
import torch
import torch.utils.data
from collections import defaultdict
from datasets import build_test_loader, build_train_loader,build_cluster_loader, build_dataset, build_transforms
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch, to_device
from models.seqnet import SeqNet
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler
import torch.nn.functional as F
from models.oim2 import OIMUnsupervisedLoss
import numpy as np
from sklearn.cluster import DBSCAN
from utils.compute_dist import *


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

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)
    print("Loading data")
    transforms = build_transforms(is_train=True)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    train_loader = build_train_loader(cfg)
    cluster_loader = build_cluster_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)
    scaler = torch.cuda.amp.GradScaler()
    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    header = "init for oim"
    metric_logger = MetricLogger(delimiter="  ")

    dataset_target_train = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    # tgt_cluster_loader = build_cluster_loader(cfg, dataset_target_train)
    
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
        # embeddings_all = []
        # labels_all = []
        # for i, (images, targets) in enumerate(metric_logger.log_every(cluster_loader, cfg.DISP_PERIOD, header)):  
        #     images, targets = to_device(images, targets, device)  
        #     embeddings, labels = model.inference_embeddings(images, targets) #[B*mean, 256] [B, 1]  
        #     embeddings_all.append(embeddings)  
        #     labels_all.append(labels) 
        # embeddings_all=torch.cat(embeddings_all, dim=0)  #[N, 256]  
        # labels_all=torch.cat(labels_all, dim=0).numpy() # [N, 1]  
        # embeddings_all = F.normalize(embeddings_all, dim=1)  
        # torch.save(embeddings_all, "embeddings_all.pt")
        embeddings_all = torch.load("embeddings_all.pt").cuda()
        # dist = compute_cosine_distance(embeddings_all, embeddings_all, cuda=False) 
        dist = compute_jaccard_distance(embeddings_all, search_option=-1)
        cluster = DBSCAN(eps=0.6, min_samples=4, metric="precomputed", n_jobs=-1,)
        indexs_to_img = defaultdict(str)
        indexs_to_pid = defaultdict(int)
        labels = cluster.fit_predict(dist)


        for anno in dataset_target_train.annotations:
            for pid,index in zip(anno["pids"], anno["indexes"]):
                indexs_to_pid[index]= pid
                indexs_to_img[index] = anno["img_name"]

        for i, label in enumerate(labels):
            print(label, indexs_to_pid[i],  indexs_to_img[i])

        # num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # print(labels)
        # print("number of clusters:==============",num_clusters)
        # labels+=1
        # print(labels)
        # labels[labels==0]=5555
        # centers = defaultdict(list)
        # for embedding, label in zip(embeddings_all,labels):
        #     if label > num_clusters+1: #for outliners and labels started on 1
        #         continue
        #     centers[label].append(embedding)
        # centers = [torch.stack(centers[idx]).mean(0) for idx in sorted(centers.keys())]
        # centers = torch.stack(centers, dim=0)
        # print(centers.shape,"====")
        # #insert background as 0
        # # labels = np.insert(labels,0,0) #0,1
        # #labels remake
        # #index_to_label = {}
        # print(dist.shape)
        # box_index=0
        # annotations = []
        # for i, ann in enumerate(dataset.annotations):
        #     num_box = len(ann["boxes"])
        #     ann["pids"] = labels[box_index:box_index+num_box]
        #     box_index+=num_box
        #     if  len(np.unique(ann["pids"]))==1 and 5555 in np.unique(ann["pids"]):
        #         continue
        #     annotations.append(ann)
        # dataset.annotations = annotations
        # # print(box_index, "====", ann["pids"], "===", np.unique(ann["pids"]),ann["pids"]==[5555])
        # train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,shuffle=True,num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,pin_memory=True,drop_last=True, collate_fn=collate_fn,)
        # # oim =OIMUnsupervisedLoss(256, num_pids=centers.shape[0],
        # #         # num_cq_size=cfg.MODEL.LOSS.CQ_SIZE, 
        # #         oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
        # #         oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,num_samples=num_clusters).cuda()
        # model.roi_heads.reid_loss.lut = F.normalize(centers, dim=1).cuda()
        # # oim.labels = torch.from_numpy(labels).cuda()
        # # print(labels, oim.labels.shape)
        # # # model.roi_heads.reid_loss.lut = F.normalize(centers, dim=1).cuda()
        # # # model.roi_head.reid_loss.label = labels
        # # model.roi_heads.reid_loss=oim
        # # for i, (images, targets) in enumerate(metric_logger.log_every(cluster_loader, cfg.DISP_PERIOD, header)):
        # #     images, targets = to_device(images, targets, device)
        # #     embeddings, labels = model.inference_embeddings(images, targets)
        # #     embeddings_all.append(embeddings)
        # #     labels_all.append(labels)
        # # embeddings_all=torch.cat(embeddings_all, dim=0)
        # # labels_all=torch.cat(labels_all, dim=0)
        # # labels_all =labels_all-1
        # # # embeddings_all = F.normalize(embeddings_all, dim=1)
        # # model.roi_heads.reid_loss.ins_lut = F.normalize(embeddings_all, dim=1).cuda()
        # # model.roi_heads.reid_loss.ins_label = labels_all.cuda()
        # # print(embeddings_all.shape, labels.shape)


        # train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard, scaler)
        # lr_scheduler.step()
        

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


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
