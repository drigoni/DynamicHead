# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from .rcnn import drigoniGeneralizedRCNN

from .concept_net import ConceptNet

__all__ = ["ConceptGeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class ConceptGeneralizedRCNN(drigoniGeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        concept_fusion,
        concept_net: nn.Module,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(
                backbone=backbone,
                proposal_generator=proposal_generator,
                roi_heads=roi_heads,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                input_format=input_format,
                vis_period=vis_period,
        )
        # Concept Net
        self.concept_net = concept_net
        self.concept_fusion = concept_fusion

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        # NOTE drigoni: this code is needed for using more than one convolutional (currently 0) network in the head.
        feature_shapes = backbone.output_shape() # {'p2': ShapeSpec(channels=256, height=None, width=None, stride=4), 'p3': ShapeSpec(channels=256, height=None, width=None, stride=8), 'p4': ShapeSpec(channels=256, height=None, width=None, stride=16), 'p5': ShapeSpec(channels=256, height=None, width=None, stride=32), 'p6': ShapeSpec(channels=256, height=None, width=None, stride=64)}
        deepsets_dim = cfg.DEEPSETS.OUTPUT_DIM
        concept_fusion = cfg.CONCEPT.CONCEPT_FUSION
        if concept_fusion == "cat":
            feature_shapes = {k: ShapeSpec(channels=f.channels + deepsets_dim, height=f.height, width=f.width, stride=f.stride)
                          for k, f in feature_shapes.items()}
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
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, feature_shapes),
            "roi_heads": build_roi_heads(cfg, feature_shapes),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "concept_net": concept_net,
            "concept_fusion": cfg.CONCEPT.CONCEPT_FUSION,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # print("Features.shape for f:", {k: f.shape for k, f in features.items()})
        # Features.shape for f: {'p2': torch.Size([5, 256, 200, 320]), 'p3': torch.Size([5, 256, 100, 160]), 'p4': torch.Size([5, 256, 50, 80]), 'p5': torch.Size([5, 256, 25, 40]), 'p6': torch.Size([5, 256, 13, 20])}

        # concepts batching and tokenization
        concepts, concepts_mask = self.concept_net.preprocess_concepts(batched_inputs)
        concepts = concepts.to(self.device)
        concepts_mask = concepts_mask.to(self.device)
        # conceptnet execution
        concepts_features = self.concept_net(concepts, concepts_mask)  # [b, 150]
        concepts_features = concepts_features.unsqueeze(-1).unsqueeze(-1)       # [b, 150, 1, 1]
        # features concatenation
        if self.concept_fusion == "cat":
            features = {k: torch.cat([f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3])], dim=1)
                        for k, f in features.items()}
        elif self.concept_fusion == "mul":
            features = {k: torch.mul(f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3]))
                        for k, f in features.items()}
        elif self.concept_fusion == "add":
            features = {k: torch.add(f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3]))
                        for k, f in features.items()}
        elif self.concept_fusion == "zeros":
            concepts_features = torch.zeros_like(concepts_features, requires_grad=False)
            features = {k: torch.cat([f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3])], dim=1)
                        for k, f in features.items()}
        else:
            logger.error("Error. CONCEPT.FUSION={} not valid. ".format(self.concept_fusion))
            exit(1)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # print("Features.shape for f:", {k: f.shape for k, f in features.items()})
        # Features.shape for f: {'p2': torch.Size([5, 256, 200, 320]), 'p3': torch.Size([5, 256, 100, 160]), 'p4': torch.Size([5, 256, 50, 80]), 'p5': torch.Size([5, 256, 25, 40]), 'p6': torch.Size([5, 256, 13, 20])}

        # concepts batching and tokenization
        concepts, concepts_mask = self.concept_net.preprocess_concepts(batched_inputs)
        concepts = concepts.to(self.device)
        concepts_mask = concepts_mask.to(self.device)
        # conceptnet execution
        concepts_features = self.concept_net(concepts, concepts_mask)  # [b, 150]
        concepts_features = concepts_features.unsqueeze(-1).unsqueeze(-1)       # [b, 150, 1, 1]
        # features concatenation
        if self.concept_fusion == "cat":
            features = {k: torch.cat([f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3])], dim=1)
                        for k, f in features.items()}
        elif self.concept_fusion == "mul":
            features = {k: torch.mul(f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3]))
                        for k, f in features.items()}
        elif self.concept_fusion == "add":
            features = {k: torch.add(f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3]))
                        for k, f in features.items()}
        elif self.concept_fusion == "zeros":
            concepts_features = torch.zeros_like(concepts_features, requires_grad=False)
            features = {k: torch.cat([f, concepts_features.repeat(1, 1, f.shape[2], f.shape[3])], dim=1)
                        for k, f in features.items()}
        else:
            logger.error("Error. CONCEPT.FUSION={} not valid. ".format(self.concept_fusion))
            exit(1)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        processed_results = drigoniGeneralizedRCNN._postprocess(instances, batched_inputs, image_size)
        return processed_results
