# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import os
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append(os.getcwd())
from reid import datasets
from reid import models
from reid.models.cm import ClusterMemory
from reid.trainers import ClusterContrastTrainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.faiss_rerank import compute_jaccard_distance
from reid.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam, RandomSampler
from datasets.build import *
from engine import evaluate_performance_twostage
start_epoch = best_mAP = 0

def get_data(name, data_dir):
    # root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    # sampler = RandomSampler(train_set)
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def build_test_loader_ps():
    transforms = build_transforms(is_train=False)
    # gallery_set = build_dataset("CUHK-SYSU", "data/cuhk_sysu", transforms, "train")
    gallery_set = build_dataset("CUHKUnsupervised", "data/CUHK-SYSU", transforms, "gallery")
    query_set = build_dataset("CUHKUnsupervised", "data/CUHK-SYSU", transforms, "query")
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return gallery_loader, query_loader

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model = create_model(args)
    gallery_loader, query_loader =build_test_loader_ps()
    # model.load_state_dict(torch.load("/home/lzy/UN_PS/UnPsDebug/instance_8_test_0.7_cuhk/model_best.pth.tar")['state_dict'], strict=True)

    # features, _ = extract_features(model, test_loader)
    # query_features = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.query], 0)
    # gallery_features = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.gallery], 0)
    # ret = evaluate_performance_twostage(
    #     gallery_loader,
    #     query_loader,
    #     gallery_features, 
    #     query_features,
    #     "cuda",
    #     use_cbgm=False,
    # )

    # exit(0)



    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)





    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
                cluster_tight = DBSCAN(eps=eps-0.02, min_samples=4, metric='precomputed', n_jobs=-1)
                cluster_loose = DBSCAN(eps=eps+0.02, min_samples=4, metric='precomputed', n_jobs=-1)

            pseudo_labels = cluster.fit_predict(rerank_dist)


        #     pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        #     pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)

        #     num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        #     num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        #     num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)
        #     # # select & cluster images as training set of this epochs
        #     # pseudo_labels = cluster.fit_predict(rerank_dist)
        # # generate new dataset and calculate cluster centers
        #     def generate_pseudo_labels(cluster_id, num):
        #         labels = []
        #         outliers = 0
        #         for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
        #             if id!=-1:
        #                 labels.append(id)
        #             else:
        #                 labels.append(num+outliers)
        #                 outliers += 1
        #         return torch.Tensor(labels).long()

        #     pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        #     pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
        #     pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

        #     N = pseudo_labels.size(0)
        #     label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        #     label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        #     label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

        #     R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
        #     R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
        #     assert((R_comp.min()>=0) and (R_comp.max()<=1))
        #     assert((R_indep.min()>=0) and (R_indep.max()<=1))

        #     cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        #     cluster_img_num = collections.defaultdict(int)
        #     for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
        #         cluster_R_comp[label.item()].append(comp.item())
        #         cluster_R_indep[label.item()].append(indep.item())
        #         cluster_img_num[label.item()]+=1

        #     cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        #     cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        #     cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
        #     if (epoch==0):
        #         indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

        #     pseudo_labeled_dataset = []
        #     outliers = 0
        #     for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
        #         indep_score = cluster_R_indep[label.item()]
        #         comp_score = R_comp[i]
        #         if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()])):
        #             pseudo_labeled_dataset.append((fname,label.item(),cid))
        #         else:
        #             # continue
        #             pseudo_labeled_dataset.append((fname,len(cluster_R_indep)+outliers,cid))
        #             pseudo_labels[i] = len(cluster_R_indep)+outliers
        #             outliers+=1



            num_clusters = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            outliners = 0
            for index, la in enumerate(pseudo_labels):
                if la == -1:
                    pseudo_labels[index] = num_clusters+outliners
                    outliners +=1

        
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)

        del cluster_loader, features

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
        print('==> Statistics for outliers: {}'.format(sum(pseudo_labels==-1)))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            torch.save(memory, "memory_feature_cuhk_0.7.pt")
            torch.save(pseudo_labels, "pseudo_labels_cuhk_0.7.pt")
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': 1,
            }, True, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))


        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            features, _ = extract_features(model, test_loader)
            query_features = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.query], 0)
            gallery_features = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.gallery], 0)
            ret = evaluate_performance_twostage(
                gallery_loader,
                query_loader,
                gallery_features, 
                query_features,
                "cuda",
                use_cbgm=False,
            )
            mAP = ret["mAP"]
            is_best = (mAP > best_mAP)
        #     mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
        #     is_best = (mAP > best_mAP)
            # if is_best:
                # torch.save(memory, "memory_feature.pt")
                # torch.save(pseudo_labels, "pseudo_labels.pt")
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    features, _ = extract_features(model, test_loader)
    query_features = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.query], 0)
    gallery_features = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.gallery], 0)
    ret = evaluate_performance_twostage(
        gallery_loader,
        query_loader,
        gallery_features, 
        query_features,
        "cuda",
        use_cbgm=False,
    )

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    main()