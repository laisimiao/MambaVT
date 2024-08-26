import math
from collections import deque

from lib.models.mambatrack import build_mambatrack_motion
from lib.test.tracker.basetracker import BaseTracker
import torch

# from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d, hann2d_bias
from lib.train.data.processing_utils import sample_target, transform_image_to_crop, transform_image_to_crop_batch
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box


# from lib.utils.ce_utils import generate_mask_cond


class MambaTrackMotion(BaseTracker):
    def __init__(self, params, dataset_name, pre_num=None):
        super(MambaTrackMotion, self).__init__(params)
        network = build_mambatrack_motion(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.stride = self.cfg.MODEL.BACKBONE.STRIDE if 'patch8' not in self.cfg.MODEL.BACKBONE.TYPE else 16
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.stride
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = 0
        print(f"Tracking: debug: {self.debug} | use_visdom: {self.use_visdom}")
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                # if not os.path.exists(self.save_dir):
                #     os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        # prepare motion information
        self.bins = self.cfg.MODEL.BINS
        self.coor_range = self.cfg.MODEL.RANGE
        self.magic_num = (self.coor_range - 1) * 0.5
        # Set the prenum
        
        if hasattr(self.cfg.TEST.TEST_PRE_NUM, dataset_name):
            self.pre_num = self.cfg.TEST.TEST_PRE_NUM[dataset_name]
        else:
            self.pre_num = 1

        # self.pre_num = self.cfg.TEST.TEST_PRE_NUM if pre_num is None else pre_num
        self.pre_motion_search = deque(maxlen=self.pre_num)  # 存的是在raw image level上的历史目标位置信息

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template
            self.memory_frames = [template.tensors]

        # save states
        for i in range(self.pre_num):
            self.pre_motion_search.append(info['init_bbox'].copy())  # info['init_bbox'].copy()本身就是一个列表

        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        # 转化历史帧坐标到当前坐标系下
        box_out_i = transform_image_to_crop_batch(torch.Tensor(self.pre_motion_search), torch.Tensor(self.state),
                                                  resize_factor,
                                                  torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                  normalize=True)  # xyxy
        # [batch,pre_motion_num,4]
        seqs_input = box_out_i.unsqueeze(0).to(search.tensors)

        # --------- select memory frames ---------
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
        else:
            template_list = self.select_memory_frames()
        # --------- select memory frames ---------

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            # print(f"seqs_input: {self.pre_motion_search}")
            out_dict = self.network.forward(template=template_list, search=[x_dict.tensors], search_pre_coors=[seqs_input])
            if isinstance(out_dict, list):
                out_dict = out_dict[-1]

        # add hann windows
        pred_score_map = out_dict['score_map']
        # if self.add_motion_pred:
        #     motion_pred_bbox = out_dict["motion_bbox"].squeeze().cpu() * self.feat_sz  # (1, 4) xyxy normalized
        #     cx = (motion_pred_bbox[0] + motion_pred_bbox[2]) / 2
        #     cy = (motion_pred_bbox[1] + motion_pred_bbox[3]) / 2
        #     ctr_point = torch.tensor([cx, cy]).long()
        #     # hann2d return [1, 1, feat_sz, feat_sz]
        #     self.motion_window = hann2d_bias(torch.tensor([self.feat_sz, self.feat_sz]).long(), ctr_point=ctr_point).cuda()

        response = self.output_window * pred_score_map
        # if self.add_motion_pred:
        #     response = self.motion_window * response
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'],
                                                                out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # -------------- vis template memories and current frame results ------------ #
        if self.debug:
            pred_iou = out_dict['pred_iou'] if 'pred_iou' in out_dict else None
            from lib.utils.box_ops import box_cxcywh_to_xyxy
            z_size = self.params.template_size
            x_size = self.params.search_size
            mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()
            std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()

            canvas = np.zeros((x_size * 2, z_size * len(template_list) + x_size, 3), dtype=np.uint8)
            tmp_z = [((i * std + mean) * 255.0).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8) for i in template_list]  # (128, 128, 6)
            tmp_x = ((search.tensors * std + mean) * 255.0).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (256, 256, 6)

            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).mean(dim=0) * x_size
            pred_boxes_xyxy = list(map(int, pred_boxes_xyxy.tolist()))
            x1, y1, x2, y2 = pred_boxes_xyxy
            image_BGR = cv2.cvtColor(tmp_x[:, :, :3], cv2.COLOR_RGB2BGR)
            image_DTE = cv2.cvtColor(tmp_x[:, :, 3:], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            cv2.rectangle(image_DTE, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            if pred_iou is not None:
                cv2.putText(image_BGR, 'iou:' + str(round(pred_iou.item(), 2)), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image_BGR, 'mxs:' + str(round(max_score, 3)), (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            image_x = np.vstack([image_BGR, image_DTE])

            for i in range(len(tmp_z)):
                z_BGR = cv2.cvtColor(tmp_z[i][:, :, :3], cv2.COLOR_RGB2BGR)
                z_DTE = cv2.cvtColor(tmp_z[i][:, :, 3:], cv2.COLOR_RGB2BGR)
                canvas[:z_size * 2, i * z_size:i * z_size + z_size] = np.vstack([z_BGR, z_DTE])
            canvas[:x_size * 2, -x_size:] = image_x
            cv2.imshow('all', canvas)
            cv2.waitKey(0)

        # ------------- save memory frames ---------------------
        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        cur_frame_tensors = cur_frame.tensors
        if self.frame_id > self.cfg.TEST.MEMORY_LENGTH:
            cur_frame_tensors = cur_frame_tensors.detach().cpu()
        self.memory_frames.append(cur_frame_tensors)
        self.pre_motion_search.append(self.state)
        # 当队列中的元素数量达到 maxlen 时，如果继续向队列中添加元素，最左边（即队列的开始位置）的元素会被自动移除
        # ------------- save memory frames ---------------------

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames = []

        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)

        return select_frames

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return MambaTrackMotion
