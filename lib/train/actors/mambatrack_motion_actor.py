from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class MambaTrackMotionActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_pre_coords: (N_s, batch, pre_motion_num, 4) 
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['template_images']) == 1
        # assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_list = []
        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)

        search_pre_coors = []
        for i in range(self.settings.num_search):
            # 统一集中到最后一帧坐标系上，并且是xyxy format 经过归一化
            search_pre_coor_i = data['search_pre_coords'][i].view(-1, *data['search_pre_coords'].shape[2:])  # (batch, pre_motion_num, 4)
            search_pre_coors.append(search_pre_coor_i)

        out_dict = self.net(template=template_list,
                            search=search_list,
                            search_pre_coors=search_pre_coors,
                            )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # currently only support the type of pred_dict is list
        if not isinstance(pred_dict, list):
            pred_dict = [pred_dict]

        magic_num = (self.cfg.MODEL.RANGE - 1) * 0.5
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()  # 定义 0 tensor，并指定GPU设备
        # search_real_anno -> search_anno in processing
        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                                 self.cfg.MODEL.BACKBONE.STRIDE)

        for i in range(len(pred_dict)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4) normalized
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(-1 * magic_num, 1 + magic_num)
            # (B,4) --> (B,1,4) --> (B,N,4)

            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss

            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss

            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss

            # compute  loss
            if 'pred_iou' in pred_dict[i]:
                conf_loss = self.objective['iou'](pred_dict[i]['pred_iou'], iou)
                loss_dict['iou'] = conf_loss
            else:
                pass

            if 'motion_bbox' in pred_dict[i]:
                pred_motion_bbox = box_cxcywh_to_xyxy(pred_dict[i]['motion_bbox']).view(-1, 4)
                # giou_loss_motion, _ = self.objective['giou'](pred_motion_bbox, gt_boxes_vec)
                l1_loss_motion = self.objective['l1'](pred_motion_bbox, gt_boxes_vec)  # (BN,4) (BN,4)
                # loss_dict['giou_motion'] = giou_loss_motion
                loss_dict['l1_motion'] = l1_loss_motion
            else:
                pass 

            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            total_loss += loss

            if return_status:
                # status for log
                status = {}

                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                          f"{i}frame_Loss/giou": giou_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/location": location_loss.item(),
                          f"{i}frame_IoU": mean_iou.item()}
                if 'pred_iou' in pred_dict[i]:
                    status.update({f"{i}frame_Loss/confidence": conf_loss.item()})
                if 'motion_bbox' in pred_dict[i]:
                    status.update({f"{i}frame_Loss/l1_motion": l1_loss_motion.item()})

                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss