from __future__ import print_function, absolute_import
import os.path as osp
import glob
import os
import re
from ..utils.data import BaseImageDataset

class CUHK(BaseImageDataset):
    def __init__(self, data_dir = 'data_dir', verbose = True):
        super(CUHK, self).__init__()
        self.dataset_dir = osp.join(data_dir, 'cuhk_sysu')
        print(data_dir)
        self.train_dir = osp.join(self.dataset_dir, 'crop_train_imgs')
        self.query_dir = osp.join(self.dataset_dir, 'crop_query_imgs')
        self.gallery_dir = osp.join(self.dataset_dir, 'crop_gallery_imgs')
        self.img_pid = {}
        for i, img in enumerate(glob.glob(osp.join(self.dataset_dir, 'gallery','*jpg'))):
            self.img_pid[img.split('/')[-1]]=i
        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False,gallery=False)
        gallery = [] #self._process_dir(self.gallery_dir, relabel=False,gallery=True)
        if verbose:
            print("=> prw loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, data_dir, relabel=True, gallery=False):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))  #
        pattern = re.compile(r'(\d+)_(\d+)')
        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid >=5000:
        #         continue
        #     if pid == -1: continue  # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # if not gallery:
        #     for label, pid in enumerate(pid_container):
        #         print(label,"=========",pid)
        dataset = []
        for img_path in img_paths:
            pid, index = map(int, pattern.search(img_path).groups())
            if gallery:
                dataset.append((img_path, pid, -1))
                continue
            dataset.append((img_path, pid, 1))
        return dataset
    def _process_dir_train(self, data_dir, relabel=True, gallery=False):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))  #
        pattern = re.compile(r'(\d+)_(\d+)')
        pid_container = set()
        for img_path in img_paths:
            pid = int(img_path[-10:-4])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in img_paths:
            pid = int(img_path[-10:-4])
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, 1))
        return dataset
