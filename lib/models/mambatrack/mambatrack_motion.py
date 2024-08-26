"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mambatrack.videomamba_motion import videomamba_base, videomamba_middle, videomamba_small, videomamba_tiny
from lib.models.mambatrack.vit import vit_base_patch16_224


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, output_dim, h, BN=False):
        super().__init__()
        self.num_layers = len(h)
        # h = [(input_dim//2**(i+1)) for i in range(num_layers - 1)]
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP2(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MambaTrackMotion(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, backbone, box_head, cfg, aux_loss=False, head_type="CORNER", patch8=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.cfg = cfg
        self.concat_mode = cfg.MODEL.BACKBONE.CONCAT_MODE
        self.add_cls_token = cfg.MODEL.BACKBONE.ADD_CLS_TOKEN
        self.add_motion_pred = cfg.MODEL.BACKBONE.ADD_MOTION_PRED

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz) if not patch8 else int(box_head.feat_sz * 2)
            self.feat_len_s = int(box_head.feat_sz ** 2) if not patch8 else int((box_head.feat_sz * 2) ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        if self.add_cls_token:
            self.iou_head = MLP(backbone.embed_dim, 1, [backbone.embed_dim//2, backbone.embed_dim//4])
        
        if self.add_motion_pred:
            self.motion_head = MLP(backbone.embed_dim, 1, [256, 128, 64, 32])

    def forward(self, template: torch.Tensor, search: torch.Tensor, search_pre_coors: torch.Tensor, return_x=False):
        # search_pre_coors: [batch,pre_motion_num,4] * N_s 表示的最后一帧坐标系上，并且是xyxy format 经过归一化
        assert isinstance(search, list), "The type of search is not List"
        length_motion = search_pre_coors[0].shape[1] * search_pre_coors[0].shape[2]

        out_dict = []
        for i in range(len(search)):
            x = self.backbone(z=template.copy(), x=search[i],
                              search_pre_coor=search_pre_coors[i], concat_mode=self.concat_mode)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            num_motion_pred = 4 if self.add_motion_pred else 0

            if self.add_cls_token:
                iou_feat = feat_last[:, :1]
                pred_iou = self.iou_head(iou_feat.squeeze()).squeeze()
                img_embed = feat_last[:, 1:-(length_motion+num_motion_pred)]
            else:
                img_embed = feat_last[:, :-(length_motion+num_motion_pred)] 

            if self.concat_mode == "tsts":
                # t_rgb, s_rgb, t_X, s_X, self.motion_token.expand(B, -1, -1)
                all_length = img_embed.shape[1]
                s_rgb = img_embed[:, all_length//2-self.feat_len_s : all_length//2]
                s_X = img_embed[:, all_length-self.feat_len_s : all_length]
            elif self.concat_mode == "ttss" or self.concat_mode == "crosst":
                s_rgb = img_embed[:, -2 * self.feat_len_s:-self.feat_len_s]
                s_X = img_embed[:, -self.feat_len_s:]
            else:
                raise NotImplementedError(f'no implement such {self.concat_mode} mode!')

            out = self.forward_head(s_rgb, s_X, None)

            if self.add_motion_pred:
                motion_bbox = _sigmoid(self.motion_head(feat_last[:, -num_motion_pred:]).squeeze())  # (B, 4)
                motion_bbox = box_xyxy_to_cxcywh(motion_bbox)
                out["motion_bbox"] = motion_bbox

            if self.add_cls_token:
                out["pred_iou"] = pred_iou
            if return_x:
                out["feat_last"] = feat_last

            out_dict.append(out)

        return out_dict


    def forward_head(self, s_rgb, s_X, gt_score_map=None):
        """
        s_rgb, s_X: output embeddings of the backbone, it can be (B, Hx*Wx, C)
        """
        srgb = (s_rgb.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = srgb.size()
        s_rgb = srgb.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        sX = (s_X.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = sX.size()
        s_X = sX.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        opt_feat = s_rgb + s_X

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mambatrack_motion(cfg, training=True):

    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained = ''

    # ---------------------------------------
    # default resolution is 224
    if cfg.MODEL.BACKBONE.TYPE == 'videomamba_tiny_576':
        backbone = videomamba_tiny(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=576,
                                   add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                   bins = cfg.MODEL.BINS,
                                   coor_range = cfg.MODEL.RANGE,
                                   pre_num = cfg.DATA.PRE_MOTION_NUM,
                                   add_motion_pred = cfg.MODEL.BACKBONE.ADD_MOTION_PRED,
                                   prompt_embed_type = cfg.MODEL.BACKBONE.PROMPT_EMBED_TYPE
                                   )
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_small_576':
        backbone = videomamba_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=576,
                                    add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                    bins=cfg.MODEL.BINS,
                                    coor_range=cfg.MODEL.RANGE,
                                    pre_num=cfg.DATA.PRE_MOTION_NUM,
                                    add_motion_pred = cfg.MODEL.BACKBONE.ADD_MOTION_PRED,
                                    prompt_embed_type = cfg.MODEL.BACKBONE.PROMPT_EMBED_TYPE
                                    )
        hidden_dim = backbone.embed_dim
    
    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_middle':
        backbone = videomamba_middle(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                     add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                     bins=cfg.MODEL.BINS,
                                     coor_range=cfg.MODEL.RANGE,
                                     pre_num=cfg.DATA.PRE_MOTION_NUM,
                                     add_motion_pred = cfg.MODEL.BACKBONE.ADD_MOTION_PRED,
                                     prompt_embed_type = cfg.MODEL.BACKBONE.PROMPT_EMBED_TYPE
                                     )
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_middle_576':
        backbone = videomamba_middle(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, img_size=576,
                                     add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                     bins=cfg.MODEL.BINS,
                                     coor_range=cfg.MODEL.RANGE,
                                     pre_num=cfg.DATA.PRE_MOTION_NUM,
                                     add_motion_pred = cfg.MODEL.BACKBONE.ADD_MOTION_PRED,
                                     prompt_embed_type = cfg.MODEL.BACKBONE.PROMPT_EMBED_TYPE
                                     )
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'videomamba_base':
        backbone = videomamba_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                   add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                   bins=cfg.MODEL.BINS,
                                   coor_range=cfg.MODEL.RANGE,
                                   pre_num=cfg.DATA.PRE_MOTION_NUM,
                                   add_motion_pred = cfg.MODEL.BACKBONE.ADD_MOTION_PRED,
                                   prompt_embed_type = cfg.MODEL.BACKBONE.PROMPT_EMBED_TYPE
                                   )
        hidden_dim = backbone.embed_dim
    # ---------------------------------------
    elif cfg.MODEL.BACKBONE.TYPE == 'ostrack_base':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim



    else:
        raise NotImplementedError

    if 'videomamba' in cfg.MODEL.BACKBONE.TYPE or 'vim' in cfg.MODEL.BACKBONE.TYPE:
        backbone.interpolate_pos_embed(cfg=cfg)
    elif 'ostrack' in cfg.MODEL.BACKBONE.TYPE:
        backbone.finetune_track(cfg, 1)

    box_head = build_box_head(cfg, hidden_dim, patch8="patch8" in cfg.MODEL.BACKBONE.TYPE)

    model = MambaTrackMotion(
        backbone,
        box_head,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        patch8="patch8" in cfg.MODEL.BACKBONE.TYPE
    )

    if 'MambaTrack' in cfg.MODEL.PRETRAIN_FILE and training:  # stage1 checkpoint
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained stage1 from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained stage1 missing_keys: ', missing_keys)
        print('Load pretrained stage1 unexpected_keys: ', unexpected_keys)

    return model

if __name__ == '__main__':
    import importlib

    config_module = importlib.import_module("lib.config.%s.config" % 'mambatrack')
    cfg = config_module.cfg
    config_module.update_config_from_file('/home/lz/PycharmProjects/dev/MambaTrack/experiments/mambatrack/videomambab_finetune_256_16x2_ep40_1t1s_ttss_nofuse.yaml')
    ostrack = build_mambatrack_motion()(cfg, training=True)
    # print(mambatrack.backbone)
