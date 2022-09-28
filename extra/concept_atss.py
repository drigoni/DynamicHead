#!/usr/bin/env python
"""
Created on 18/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: 
"""

import logging
import math
from typing import Dict, List
import torch
from torch import Tensor, nn
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.comm import get_world_size
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .sigmoid_focal_loss import SigmoidFocalLoss
from .concept_net import ConceptNet
from .atss import ATSS, reduce_sum, concat_box_prediction_layers, permute_and_flatten, Scale

__all__ = ["CATSS"]

logger = logging.getLogger(__name__)

INF = 1e8

@META_ARCH_REGISTRY.register()
class CATSS(ATSS):
    """
    Implement ATSS based on https://github.com/sfzhang15/ATSS
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        box_reg_loss_weight=2.0,
        pre_nms_thresh=0.05,
        pre_nms_top_n=1000,
        nms_thresh=0.6,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        input_format="BGR",
        anchor_aspect_ratio,
        anchor_topk,
        concept_fusion,
        concept_net: nn.Module,
    ):
        super().__init__(
                backbone=backbone,
                head=head, 
                head_in_features=head_in_features, 
                anchor_generator=anchor_generator, 
                box2box_transform=box2box_transform, 
                anchor_matcher=anchor_matcher, 
                num_classes=num_classes, 
                focal_loss_alpha=focal_loss_alpha, 
                focal_loss_gamma=focal_loss_gamma, 
                box_reg_loss_weight=box_reg_loss_weight, 
                pre_nms_thresh=pre_nms_thresh, 
                pre_nms_top_n=pre_nms_top_n, 
                nms_thresh=nms_thresh, 
                max_detections_per_image=max_detections_per_image, 
                pixel_mean=pixel_mean, 
                pixel_std=pixel_std, 
                input_format=input_format, 
                anchor_aspect_ratio=anchor_aspect_ratio, 
                anchor_topk=anchor_topk, 
        )
        # Concept Net
        self.concept_net = concept_net
        self.concept_fusion = concept_fusion

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.ATSS.IN_FEATURES]
        # TODO drigoni: this code is needed for using more than one convolutional (currently 0) network in the head.
        deepsets_dim = cfg.DEEPSETS.OUTPUT_DIM
        concept_fusion = cfg.CONCEPT.CONCEPT_FUSION
        if concept_fusion == "cat":
            feature_shapes = [ShapeSpec(channels=f.channels + deepsets_dim, height=f.height, width=f.width, stride=f.stride)
                          for f in feature_shapes]
        elif concept_fusion == "mul":
            feature_shapes = [ShapeSpec(channels=f.channels, height=f.height, width=f.width, stride=f.stride)
                          for f in feature_shapes]
        elif concept_fusion == "add":
            feature_shapes = [ShapeSpec(channels=f.channels, height=f.height, width=f.width, stride=f.stride)
                          for f in feature_shapes]
        elif concept_fusion == "zeros":
            feature_shapes = [ShapeSpec(channels=f.channels + deepsets_dim, height=f.height, width=f.width, stride=f.stride)
                          for f in feature_shapes]
        else:
            logger.error("Error. CONCEPT.FUSION={} not valid. ".format(concept_fusion))
            exit(1)
        concept_net = ConceptNet(cfg)
        head = CATSSHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ATSS.BBOX_REG_WEIGHTS),
            "anchor_matcher": Matcher(
                cfg.MODEL.ATSS.IOU_THRESHOLDS,
                cfg.MODEL.ATSS.IOU_LABELS,
                allow_low_quality_matches=True,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.ATSS.NUM_CLASSES,
            "head_in_features": cfg.MODEL.ATSS.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.ATSS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.ATSS.FOCAL_LOSS_GAMMA,
            "box_reg_loss_weight": cfg.MODEL.ATSS.REG_LOSS_WEIGHT,
            # Inference parameters:
            "pre_nms_thresh": cfg.MODEL.ATSS.INFERENCE_TH,
            "pre_nms_top_n": cfg.MODEL.ATSS.PRE_NMS_TOP_N,
            "nms_thresh": cfg.MODEL.ATSS.NMS_TH,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "input_format": cfg.INPUT.FORMAT,
            "anchor_aspect_ratio": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "anchor_topk": cfg.MODEL.ATSS.TOPK,
            "concept_net": concept_net,
            "concept_fusion": cfg.CONCEPT.CONCEPT_FUSION,
        }

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        # images.tensor -> [b, rgb, w, h]
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]  # List[Tensors] five element with shape [b, 256, w, h]
        anchors = self.anchor_generator(features)

        # concepts batching and tokenization
        concepts, concepts_mask = self.concept_net.preprocess_concepts(batched_inputs)
        concepts = concepts.to(self.device)
        concepts_mask = concepts_mask.to(self.device)
        # conceptnet execution
        concepts_features = self.concept_net(concepts, concepts_mask)  # [b, 150]
        concepts_features = concepts_features.unsqueeze(-1).unsqueeze(-1)       # [b, 150, 1, 1]
        # features concatenation
        if self.concept_fusion == "cat":
            features = [torch.cat([f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3])], dim=1)
                        for f in features]
        elif self.concept_fusion == "mul":
            features = [torch.mul(f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3]))
                    for f in features]
        elif self.concept_fusion == "add":
            features = [torch.add(f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3]))
                    for f in features]
        elif self.concept_fusion == "zeros":
            concepts_features = torch.zeros_like(concepts_features, requires_grad=False)
            features = [torch.cat([f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3])], dim=1)
                    for f in features]
        else:
            logger.error("Error. CONCEPT.FUSION={} not valid. ".format(self.concept_fusion))
            exit(1)

        pred_logits, pred_anchor_deltas, pred_centers, pred_features = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centers)

            return losses
        else:
            res_instances = self.inference(anchors, pred_logits, pred_anchor_deltas, pred_centers, pred_features, images.image_sizes)
            if torch.jit.is_scripting():
                return res_instances
            processed_results = []
            for results_per_image, input_per_image, image_size in \
                    zip(res_instances, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results


class CATSSHead(torch.nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(CATSSHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ATSS.NUM_CLASSES
        num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS)
        use_gn = cfg.MODEL.ATSS.USE_GN
        deepsets_channel = cfg.DEEPSETS.OUTPUT_DIM
        concept_fusion = cfg.CONCEPT.CONCEPT_FUSION
        if concept_fusion == "cat":
            channels = cfg.MODEL.ATSS.CHANNELS + deepsets_channel
        elif concept_fusion == "mul":
            channels = cfg.MODEL.ATSS.CHANNELS
        elif concept_fusion == "add":
            channels = cfg.MODEL.ATSS.CHANNELS
        elif concept_fusion == "zeros":
            channels = cfg.MODEL.ATSS.CHANNELS + deepsets_channel
        else:
            logger.error("Error. CONCEPT.FUSION={} not valid. ".format(concept_fusion))
            exit(1)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if cfg.MODEL.ATSS.NUM_CONVS > 0:
            cls_tower = []
            bbox_tower = []
            for i in range(cfg.MODEL.ATSS.NUM_CONVS):
                cls_tower.append(
                    nn.Conv2d(
                        in_channels if i==0 else channels,
                        channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    )
                )
                if use_gn:
                    cls_tower.append(nn.GroupNorm(32, channels))

                cls_tower.append(nn.ReLU())

                bbox_tower.append(
                    nn.Conv2d(
                        in_channels if i == 0 else channels,
                        channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    )
                )
                if use_gn:
                    bbox_tower.append(nn.GroupNorm(32, channels))

                bbox_tower.append(nn.ReLU())

            self.add_module('cls_tower', nn.Sequential(*cls_tower))
            self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        else:
            self.cls_tower = None
            self.bbox_tower = None

        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialization
        if cfg.MODEL.ATSS.NUM_CONVS>0:
            for modules in [self.cls_tower, self.bbox_tower]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        features = []
        for l, feature in enumerate(x):
            features.append(feature)
            if self.cls_tower is None:
                cls_tower = feature
            else:
                cls_tower = self.cls_tower(feature)

            if self.bbox_tower is None:
                box_tower = feature
            else:
                box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness, features
