import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class COESOT(BaseVideoDataset):
    def __init__(self, root=None, split='train', dtype='rgbrgb', image_loader=jpeg4py_loader_w_failsafe,):

        root = env_settings().coesot_train_dir if root is None else root
        super().__init__('COESOT', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        dir_list = [i for i in os.listdir(os.path.join(self.root))
                    if os.path.isdir(os.path.join(self.root, i))]
        return dir_list

    def get_name(self):
        return 'coesot_' + self.split

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        bbox_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(bbox_path)

        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 4.0) & (bbox[:, 3] > 4.0)
        visible = valid.clone().byte().bool()
        return {'bbox': bbox, 'valid': valid, 'visible': visible, }

    def _get_frame_path(self, seq_path, frame_id):
        seq_name = seq_path.split('/')[-1]
        aps_dir = os.path.join(seq_path, seq_name + '_aps')
        dvs_dir = os.path.join(seq_path, seq_name + '_dvs')
        if os.path.exists(os.path.join(aps_dir, 'frame{:04}.png'.format(frame_id))):
            vis_path = os.path.join(aps_dir, 'frame{:04}.png'.format(frame_id))
        else:
            vis_path = os.path.join(aps_dir, 'frame{:04}.bmp'.format(frame_id))

        if os.path.exists(os.path.join(dvs_dir, 'frame{:04}.bmp'.format(frame_id))):
            event_path = os.path.join(dvs_dir, 'frame{:04}.bmp'.format(frame_id))
        else:
            event_path = os.path.join(dvs_dir, 'frame{:04}.png'.format(frame_id))

        return vis_path, event_path

    def _get_frame(self, seq_path, frame_id):
        color_path, event_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(color_path, event_path, dtype=self.dtype, depth_clip=False)
        return img  # (h,w,6)


    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   })

        return frame_list, anno_frames, object_meta

