# Modified for SALA (Size-Aware Label Assignment)
# Paper: Pillar-SALNet
import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in model_cfg.ANCHOR_GENERATOR_CONFIG]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        self.anchor_size_map = {}

        for config in model_cfg.ANCHOR_GENERATOR_CONFIG:
            cls_name = config['class_name']
            self.matched_thresholds[cls_name] = config['matched_threshold']
            self.unmatched_thresholds[cls_name] = config['unmatched_threshold']
            self.anchor_size_map[cls_name] = np.array(config['anchor_sizes'][0], dtype=np.float32)

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)

        # =========================================================
        # SALA: Size-Aware Adaptive Label Assignment
        # 根据 GT 尺寸与该类 anchor 先验尺寸的一致性，动态调整 IoU 阈值
        # 一致性差 -> 阈值适度放宽
        # =========================================================
        self.sala_enabled = anchor_target_cfg.get('SALA_ENABLED', False)
        self.sala_alpha = 0.20   # matched_threshold 最大下调比例
        self.sala_beta = 0.10    # unmatched_threshold 最大下调比例
        self.sala_max_dev = 0.35 # log-space deviation 截断
        self.sala_min_gap = 0.05 # unmatched 始终小于 matched

    def _get_sala_thresholds(self, gt_boxes, class_name, base_matched, base_unmatched, device):
        """
        gt_boxes: (M, 7)
        return:
            gt_matched_thresholds: (M,)
            gt_unmatched_thresholds: (M,)
        """
        num_gt = gt_boxes.shape[0]
        if (not self.sala_enabled) or num_gt == 0 or class_name not in self.anchor_size_map:
            gt_matched = torch.full((num_gt,), float(base_matched), dtype=torch.float32, device=device)
            gt_unmatched = torch.full((num_gt,), float(base_unmatched), dtype=torch.float32, device=device)
            return gt_matched, gt_unmatched

        prior_dims = torch.tensor(self.anchor_size_map[class_name], dtype=torch.float32, device=device).view(1, 3)
        gt_dims = gt_boxes[:, 3:6].clamp(min=1e-3)

        # 尺寸偏差：加权 log-space 偏差（L, W, H）
        dim_weights = torch.tensor(
            [0.5, 0.3, 0.2], dtype=torch.float32, device=device
        ).view(1, 3)

        dev = torch.sum(
            dim_weights * torch.abs(torch.log(gt_dims / prior_dims)),
            dim=1
        )  # (M,)

        dev_ratio = torch.clamp(dev / self.sala_max_dev, min=0.0, max=1.0)

        # 双向自适应：
        # dev_ratio < 0.5 时，目标更接近先验，适当提高阈值
        # dev_ratio > 0.5 时，目标偏离先验，适当降低阈值
        matched_adjust = self.sala_alpha * (dev_ratio - 0.5) * 2.0
        unmatched_adjust = self.sala_beta * (dev_ratio - 0.5) * 2.0

        gt_matched = base_matched * (1.0 - matched_adjust)
        gt_unmatched = base_unmatched * (1.0 - unmatched_adjust)

        gt_matched = torch.clamp(
            gt_matched,
            min=float(base_matched) * 0.75,
            max=float(base_matched) * 1.10
        )
        gt_unmatched = torch.clamp(
            gt_unmatched,
            min=float(base_unmatched) * 0.75,
            max=float(base_unmatched) * 1.10
        )

        # 保证 unmatched < matched
        gt_unmatched = torch.minimum(gt_unmatched, gt_matched - self.sala_min_gap)
        return gt_matched, gt_unmatched

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]

        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 0:
                    cls_idx = (cur_gt_classes.cpu().numpy() - 1).astype(np.int64)
                    cls_idx = np.clip(cls_idx, 0, len(self.class_names) - 1)
                    mask_np = (self.class_names[cls_idx] == anchor_class_name)
                    mask = torch.from_numpy(np.asarray(mask_np, dtype=np.bool_)).to(cur_gt_classes.device)
                else:
                    mask = torch.zeros((0,), dtype=torch.bool, device=cur_gt_classes.device)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                else:
                    anchors = anchors.view(-1, anchors.shape[-1])

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=cur_gt_classes[mask],
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name],
                    class_name=anchor_class_name
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*anchors.shape[:-1]) for t, anchors in zip(target_list, all_anchors)],
                    'box_reg_targets': [t['box_reg_targets'].view(*anchors.shape[:-1], self.box_coder.code_size)
                                        for t, anchors in zip(target_list, all_anchors)],
                    'reg_weights': [t['reg_weights'].view(*anchors.shape[:-1]) for t, anchors in zip(target_list, all_anchors)]
                }

                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(
                    target_dict['box_cls_labels'], dim=-1
                ).view(-1)

                target_dict['reg_weights'] = torch.cat(
                    target_dict['reg_weights'], dim=-1
                ).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = torch.stack(bbox_targets, dim=0)
        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)

        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights
        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45,
                              class_name=None):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = anchors.new_zeros((num_anchors,), dtype=torch.int32)
        gt_ids = anchors.new_zeros((num_anchors,), dtype=torch.int32) - 1

        gt_matched_thresholds, gt_unmatched_thresholds = self._get_sala_thresholds(
            gt_boxes, class_name, matched_threshold, unmatched_threshold, anchors.device
        )

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            if self.match_height:
                anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
            else:
                anchor_by_gt_overlap = box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]

            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)
            ]

            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            matched_thr_per_anchor = gt_matched_thresholds[anchor_to_gt_argmax]
            unmatched_thr_per_anchor = gt_unmatched_thresholds[anchor_to_gt_argmax]

            pos_inds = anchor_to_gt_max >= matched_thr_per_anchor
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()

            bg_inds = (anchor_to_gt_max < unmatched_thr_per_anchor).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disable = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds), device=anchors.device)[:num_disable]
                labels[fg_inds[disable_inds]] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,), device=anchors.device)]
                labels[enable_inds] = 0
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))
        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
