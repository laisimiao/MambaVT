import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class LasHeRMotion(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype
        self.root_parent = os.path.abspath(os.path.dirname(self.root))

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'lasher_motion'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # file_path = os.path.join(ltr_path, 'data_specs', 'lasher_motion_{}.txt'.format(split))
        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        # in lasher dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")
        # ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # Note original filename is chaotic, we rename them
        rgb_frame_path = os.path.join(seq_path, 'visible', '{:06d}.jpg'.format(frame_id))  # frames start from 0
        ir_frame_path = os.path.join(seq_path, 'infrared', '{:06d}.jpg'.format(frame_id))
        return (rgb_frame_path, ir_frame_path)  # jpg jpg

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # (h,w,6)

    def _get_pre_coords(self, visible, bboxes, x_frame_id, pre_num, reverse=False):
        if reverse:
            # 这里z_id > x_id, pre 是往大的id找 (例如z_id是[8,7,3], x_id是[2,1])
            visible_ids = []
            start_id = x_frame_id + 1
            enough_visible_ids = False
            while not enough_visible_ids:
                if start_id > len(visible):
                    if len(visible_ids) == 0:
                        raise ValueError("frame_id is selected unreasonably or visible number is too small")  # TODO: return False
                    else:
                        visible_ids.append(visible_ids[-1])
                else:
                    if visible[start_id]:
                        visible_ids.append(start_id)
                    start_id += 1

                enough_visible_ids = len(visible_ids) == pre_num

            visible_ids = visible_ids[::-1]  # 从大到小
            pre_motion_bboxes = bboxes[visible_ids]
        else:
            # 这里z_id < x_id, pre 是往小的id找 (例如z_id是[1,2,3], x_id是[7,8])
            visible_ids = []
            start_id = x_frame_id - 1
            enough_visible_ids = False
            while not enough_visible_ids:
                if start_id < 0:
                    if len(visible_ids) == 0:
                        raise ValueError("frame_id is selected unreasonably or visible number is too small")  # TODO: return False
                    else:
                        visible_ids.append(visible_ids[-1])
                else:
                    if visible[start_id]:
                        visible_ids.append(start_id)
                    start_id -= 1

                enough_visible_ids = len(visible_ids) == pre_num

            visible_ids = visible_ids[::-1]  # 从小到大
            pre_motion_bboxes = bboxes[visible_ids]
        return pre_motion_bboxes

                    
    def get_frames(self, seq_id, frame_ids, anno=None, is_search=False, pre_motion_num=0, reverse=False):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        imgh, imgw = frame_list[0].shape[:2]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        if is_search and pre_motion_num>0:
            tracker_predicted_bboxes = np.loadtxt(os.path.join(self.root_parent, 'tracker_predicted', self.sequence_list[seq_id]+'.txt'), delimiter=' ')
            pre_coords = []
            anno_frames["real_anno"] = anno_frames['bbox']
            for count, f_id in enumerate(frame_ids):
                pre_coord = self._get_pre_coords(anno["visible"], tracker_predicted_bboxes, f_id, pre_motion_num, reverse=reverse)
                pre_coord = torch.tensor(pre_coord)  # np.ndarray [pre_motion_num, 4]
                pre_coords.append(pre_coord)
            anno_frames['bbox'] = [i[-1, ...].clone() for i in pre_coords]  # 这个是模拟测试，用上一帧的坐标crop搜索区域,实际的真值在real_anno里面
            anno_frames["pre_coords"] = pre_coords


        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
