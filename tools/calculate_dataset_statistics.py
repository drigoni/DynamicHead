#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import sys
from itertools import chain
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup
from detectron2.utils.visualizer import Visualizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dyhead import add_dyhead_config
from extra import add_extra_config
from extra import add_concept_config


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    add_concept_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Calculates dataset statistics.")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))

    is_concepts = False
    n_bounding_boxes = []
    n_unique_classes = []
    n_concepts = []             # list of list
    n_images = len(dicts)
    for dic in tqdm.tqdm(dicts):
        n_bounding_boxes.append(len(dic['annotations']))
        unique_class = list(set([a['category_id'] for a in dic['annotations']]))
        n_unique_classes.append(len(unique_class))
        if dic.get('concepts', None) is not None:
            is_concepts = True
            n_concepts.append(len(dic['concepts']))
    
    # print statistics
    print("Number of images:", n_images)
    print("Mean bounding boxes per image:", sum(n_bounding_boxes)/len(n_bounding_boxes))
    print("Sum bounding boxes per image:", sum(n_bounding_boxes))
    print("Max bounding boxes per image:", max(n_bounding_boxes))
    print("Min bounding boxes per image:", min(n_bounding_boxes))
    print("Mean of unique classes per image:", sum(n_unique_classes)/len(n_unique_classes))
    print("Max of unique classes per image:", max(n_unique_classes))
    print("Min of unique classes per image:", min(n_unique_classes))
    if is_concepts:
        print("Mean concepts per image:", sum(n_concepts)/len(n_concepts))
        print("Sum concepts per image:", sum(n_concepts))
        print("Max concepts per image:", max(n_concepts))
        print("Min concepts per image:", min(n_concepts))



