#!/usr/bin/env python
"""
Created on 16/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description:
"""
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import random

from detectron2.config import configurable
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog

import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
import pycocotools.mask as mask_util

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)
from .parser_EWISER import extract_COCO_concepts
from .concept_finder import ConceptFinder

logger = logging.getLogger(__name__)

class ConceptMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            use_keypoint: bool = False,
            instance_mask_format: str = "polygon",
            keypoint_hflip_indices: Optional[np.ndarray] = None,
            precomputed_proposal_topk: Optional[int] = None,
            recompute_boxes: bool = False,
            dataset_name:list,
            coco2synset: dict = None,
            apply_condition: bool = True,
            apply_condition_from_file: bool = False,
            external_concepts_folder: str = "./datasets/ewiser_concepts_COCO_valid/",
            activate_concept_generator: bool = True,
            meta_architecture = "ATSS",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk = precomputed_proposal_topk
        self.recompute_boxes = recompute_boxes
        self.dataset_name = dataset_name
        self.coco2synset = coco2synset
        self.apply_condition = apply_condition
        self.apply_condition_from_file = apply_condition_from_file
        self.external_concepts_folder = external_concepts_folder
        self.activate_concept_generator = activate_concept_generator
        self.meta_architecture = meta_architecture
        # List with all accepted concepts
        all_accepted_concepts = [val_dict['synset'] for k, val_dict in self.coco2synset.items()]
        all_accepted_concepts.extend([v for k, val_dict in self.coco2synset.items() for v in val_dict['descendants']])
        self.all_accepted_concepts = list(set(all_accepted_concepts))
        # fmt: on
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        # select current dataset
        if is_train:
            dataset_name = cfg.DATASETS.TRAIN,  # like (('coco_2017_tuning_train',),)
        else:
            dataset_name = cfg.DATASETS.TEST,   # like (('coco_2017_tuning_train',),)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "dataset_name": dataset_name,
            "apply_condition": cfg.CONCEPT.APPLY_CONDITION,
            "apply_condition_from_file": cfg.CONCEPT.APPLY_CONDITION_FROM_FILE,
            "external_concepts_folder": cfg.CONCEPT.EXTERNAL_CONCEPTS_FOLDER,
            "activate_concept_generator": cfg.CONCEPT.ACTIVATE_CONCEPT_GENERATOR,
            "meta_architecture": cfg.MODEL.META_ARCHITECTURE,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if self.meta_architecture in ["CATSS", "ConceptGeneralizedRCNN", "ConceptRetinaNet"]:
            if self.apply_condition:
                if self.apply_condition_from_file:
                    # get input concepts from external file and keep original annotations
                    image_path = dataset_dict["file_name"]
                    image_name = image_path.split('/')[-1][:-4]
                    concepts_path = "{}{}.json".format(self.external_concepts_folder, image_name)
                    concepts_uncleaned, sentences = extract_COCO_concepts(concepts_path)
                    # # NOTE: filtering concepts to consider just those seen in training
                    # concepts = []
                    # for conc in concepts_uncleaned:
                    #     if conc in self.all_accepted_concepts:
                    #         concepts.append(conc)
                    concepts = []
                    concepts_to_avoid = ['traffic_signal.n.01']
                    # this loop search for each external concept the closest synset to those of the categories.
                    # It does not uses directly the external concept.
                    for conc in concepts_uncleaned:
                        for cat_label in self.coco2synset.keys():
                            cls_descendants = self.coco2synset[cat_label]['descendants']
                            cls_synset = self.coco2synset[cat_label]['synset']
                            # note that traffic_signal.n.01 is not in the vocabulary, so for now we remove it.
                            if conc in cls_descendants and conc not in concepts_to_avoid:
                                if cls_synset not in concepts_to_avoid:
                                    concepts.append(cls_synset)
                                else:
                                    concepts.append(conc)
                    concepts = list(set(concepts))
                    if len(concepts) == 0:
                        concepts = ['entity.n.01']
                    # do not change annotations
                    # annos = annos
                    dataset_dict["concepts"] = concepts
                else:
                    # NOTE: the categories ids in the annotations is not the same of the COCO datasets.
                    # In COCO datasets there are 90 idx but only 80 are used. These ids are not contiguous, so the model defines 80 contiguous indexes.
                    # print("self.dataset", self.dataset_name) -> self.dataset (('coco_2017_tuning_train',),)
                    metaMapping = MetadataCatalog.get(self.dataset_name[0][0]).thing_dataset_id_to_contiguous_id  # from origin ids to contiguos one
                    metaMapping_reverse = {val: key for key, val in metaMapping.items()}
                    if self.activate_concept_generator:
                        # NOTE: here we generate concepts and ground truths
                        # standard object detector behaviour or conditioned one
                        list_categories = [metaMapping_reverse[ann['category_id']] for ann in annos]
                        is_conditioned = random.randint(0, len(set(list_categories))) > 0     # proportional to number of concepts
                        if is_conditioned: # this means that we select some of the annotations
                            selected_categories, concepts = ConceptFinder.sample_categories_and_concepts(list_categories, self.coco2synset, type='subset') # subset_old TODO
                            annos_filtered = [ann for ann in annos if metaMapping_reverse[ann['category_id']] in selected_categories]
                        else:
                            # we keep all the annotations and we use the standard object detector behaviour
                            annos_filtered = annos 
                            concepts = ['entity.n.01']
                    else:
                        # NOTE: the dataset need to include the 'concepts' keyword.
                        assert 'concepts' in dataset_dict, "Wrong dataset, concept not incldued!"
                        concepts = [c for c in dataset_dict.pop("concepts")]
                        annos_filtered = annos
                    annos = annos_filtered
                    dataset_dict["concepts"] = concepts
            else:
                # do not change annotations
                # annos = annos
                dataset_dict["concepts"] = ['entity.n.01']
        elif self.meta_architecture in ["ATSS", "GeneralizedRCNN", "drigoniGeneralizedRCNN", "RetinaNet", "drigoniRetinaNet"]:
            # standard object detector behaviour without concepts.
            pass
        else:
            logger.error("Error. MODEL.META_ARCHITECTURE={} not valid. ".format(self.meta_architecture))
            exit(1)

        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict




        # print("KEYS: ", dataset_dict.keys())
        # KEYS:  dict_keys(['file_name', 'height', 'width', 'image_id', 'image', 'instances'])
        # print("KEY instances: ", dataset_dict['instances'])
        # KEY instances:  Instances(num_instances=2, image_height=704, image_width=1053,
        # fields=[gt_boxes: Boxes(tensor([[3.1590e+00, 2.7624e+02, 9.2682e+02, 5.1300e+02],
        #                                   [7.1143e+01, 1.8093e-01, 3.3568e+02, 3.2483e+02]])),
        # gt_classes: tensor([79, 39])])
        return dataset_dict