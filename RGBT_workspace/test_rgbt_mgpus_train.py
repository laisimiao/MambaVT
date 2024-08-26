import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
from datetime import timedelta

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.mambatrack_mtms import MambaTrack
# from lib.test.tracker.mambatrack_mtms_maxfusescore import MambaTrack
import lib.test.parameter.mambatrack as rgbt_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time


def genConfig(seq_path, set_type):

    if set_type == 'LasHeR_Train':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')



    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='mambatrack',hp_name='',intval=128):
    if 'VTUAV' in dataset_name:
        seq_txt = seq_name.split('/')[1]
    else:
        seq_txt = seq_name
    save_name = '{}_testep{}_intval{}'.format(yaml_name, epoch, intval) + f"{hp_name}"
    # save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder) and not debug:
        os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(save_path) and not debug:
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if script_name == 'mambatrack':
        params = rgbt_params.parameters(yaml_name, epoch, debug=debug)
        mmtrack = MambaTrack(params, dataset_name)  # "GTOT" # dataset_name
        tracker = ViPT_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    # result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        cond = (frame_idx % intval == 0) and (RGB_gt[frame_idx, 2] > 0 and RGB_gt[frame_idx, 3] > 0)
        if frame_idx == 0 or cond:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            tracker.initialize(image, RGB_gt[frame_idx].tolist())  # xywh
            result[frame_idx] = np.copy(RGB_gt[frame_idx])
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))
            region = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class ViPT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        return pred_bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='mambatrack', help='Name of tracking method().')
    parser.add_argument('--yaml_name', type=str, default='videom256_32x2_ep20_7t1s_parttrain_lr04again', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR_Train', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=6, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=20, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--intval', default=128, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--hp_name', default='', type=str, help='save diff name for hp')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/media/lz/T9/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
        # seq_list = ['BlackSwan1', 'MinibusNigOcc', 'OccCar-2', 'Pool', 'RainyCar2', 'RainyPeople', 'Walking']
        # seq_list = [ 'BlackSwan1']
    elif dataset_name == 'RGBT234':
        seq_home = '/media/lz/T9/RGBT234'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'RGBT210':
        seq_home = '/media/lz/T9/RGBT210'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/home/lz/Videos/LasHeR/test/testingset'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR_Train':
        seq_home = "/home/lz/Videos/LasHeR/train/trainingset"
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

    dataset_start_time = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.hp_name, args.intval) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.hp_name, args.intval) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))))
