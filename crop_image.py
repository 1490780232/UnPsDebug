
import torch.utils.data
from datasets import build_test_loader, build_train_loader,build_cluster_loader, build_dataset, build_transforms
import tqdm
from utils.compute_dist import *
import cv2
import os
transforms = build_transforms(is_train=False)

root = "/home/lzy/UN_PS/data/PRW"
dataset = build_dataset("PRWUnsupervised", "data/PRW", transforms, "train")
annotations = dataset.annotations
for i, ann in tqdm.tqdm(enumerate(dataset.annotations)):
    boxes = ann["boxes"]
    indexes = ann["indexes"]
    img_path = ann["img_path"]
    pids =  ann["pids"]
    im = cv2.imread(img_path)
    for i_box in range(len(indexes)):
        i_person = indexes[i_box]
        x1,y1,x2,y2 = boxes[i_box]
        pid = pids[i_box]
        cv2.imwrite(os.path.join(root, "crop_train_imgs",str(i_person).zfill(6)+".jpg"), im[y1:y2,x1:x2])

dataset = build_dataset("PRWUnsupervised", "data/PRW", transforms, "gallery")
annotations = dataset.annotations
for i, ann in tqdm.tqdm(enumerate(dataset.annotations)):
    boxes = ann["boxes"]
    indexes = ann["indexes"]
    img_path = ann["img_path"]
    pids =  ann["pids"]
    im = cv2.imread(img_path)
    for i_box in range(len(indexes)):
        i_person = indexes[i_box]
        x1,y1,x2,y2 = boxes[i_box]
        pid = pids[i_box]
        cv2.imwrite(os.path.join(root, "crop_gallery_imgs",str(pid).zfill(5)+"_"+str(i_person).zfill(5)+".jpg"), im[y1:y2,x1:x2])

count =0
dataset = build_dataset("PRWUnsupervised", "data/PRW", transforms, "query")
annotations = dataset.annotations
for i, ann in tqdm.tqdm(enumerate(dataset.annotations)):
    boxes = ann["boxes"]
    # indexes = ann["indexes"]
    img_path = ann["img_path"]
    pids =  ann["pids"]
    im = cv2.imread(img_path)
    for i_box in range(len(pids)):
        x1,y1,x2,y2 = boxes[i_box]
        pid = pids[i_box]
        cv2.imwrite(os.path.join(root, "crop_query_imgs",str(pid).zfill(5)+"_"+str(count).zfill(5)+".jpg"), im[y1:y2,x1:x2])
        count+=1