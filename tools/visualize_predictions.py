#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import sys
from itertools import chain
import cv2
import tqdm
import atexit
import bisect
import multiprocessing as mp
import torch
import argparse
import glob
import numpy as np
import os
import tempfile
import time
import warnings
import tqdm
import json
import glob
import sys
import warnings
import contextlib
from pycocotools.coco import COCO
import io
import copy
import nltk
from nltk.corpus import wordnet as wn

from detectron2.utils.file_io import PathManager
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures.boxes import Boxes

warnings.filterwarnings("ignore", category=UserWarning) 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dyhead import add_dyhead_config
from extra import add_concept_config, add_extra_config, ConceptFinder
from train_net import Trainer

nltk.download('wordnet')
nltk.download('omw-1.4')



def filtering_process(pred, input_concepts, coco2synset, dataset_metadata):
    # make the pool of accepted ancestors
    all_accepted_concepts = {dataset_metadata.thing_dataset_id_to_contiguous_id[k]: val_dict['descendants'] + [val_dict['synset']] for k, val_dict in coco2synset.items() }
    
    # select the pool of accepted classes
    poll_accepted_classes = []
    for concept in input_concepts:
        for cat_id, descendants in all_accepted_concepts.items():
            if concept in descendants:
                poll_accepted_classes.append(cat_id)
    
    # filtering
    filtered_list = {
        'pred_boxes': [],
        'pred_classes': [],
    }
    assert len(pred['pred_boxes']) == len(pred['pred_classes'])
    for i in range(len(pred['pred_boxes'])):  
        current_class = pred['pred_classes'][i]
        current_box = pred['pred_boxes'][i]
        if current_class in poll_accepted_classes:
            filtered_list['pred_classes'].append(current_class)
            filtered_list['pred_boxes'].append(current_box)

    return filtered_list


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TRAIN.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        # self.model = build_model(self.cfg)
        self.model = Trainer.build_model(cfg)
        self.model.eval()
        if len(cfg.DATASETS.TRAIN):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(cfg.MODEL.WEIGHTS)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, concepts=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            concepts (list of strings): list of concepts.
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the/myothermodule. format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width, 'concepts': concepts}
            predictions = self.model([inputs])[0]
            return predictions


class ProposalExtractor(object):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, concepts=None):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            concepts (list of strings): list of concepts.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        if concepts:
            predictions = self.predictor(image, concepts)
        else:
            predictions = self.predictor(image)
        instances = predictions["instances"].to(self.cpu_device)
        results = dict()
        for k, v in instances.get_fields().items():
            if isinstance(v, Boxes):
                boxes_list = v.tensor
                results[k] = boxes_list.tolist()
            else:
                results[k] = v.tolist()

        assert len(results['features']) ==  len(results['pred_boxes']) == len(results['probs']), 'Error in the results.'
        return results


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
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("--filtering", action="store_true", help="apply postprocessing filtering")
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

    # NOTE: drigoni: add concepts to classes
    concept_finder = ConceptFinder(cfg.CONCEPT.FILE, depth=cfg.CONCEPT.DEPTH, unique=cfg.CONCEPT.UNIQUE, only_name=cfg.CONCEPT.ONLY_NAME)
    coco2synset = concept_finder.coco2synset

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    extractor = ProposalExtractor(cfg)

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ... ".format(filepath))
            vis.save(filepath)

    scale = 1.0
    train_data_loader = build_detection_train_loader(cfg)
    for batch in train_data_loader:
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            # current_image = read_image(per_image["file_name"], format="RGB")
            current_image = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()

            # check wether the concepts are used or not.
            if cfg.MODEL.META_ARCHITECTURE in ["CATSS", "ConceptGeneralizedRCNN", "ConceptRetinaNet"] and cfg.CONCEPT.APPLY_CONDITION:
                print("Using concepts: {}", per_image["concepts"])
                predictions = extractor.run_on_image(current_image, per_image["concepts"])
            else:
                predictions = extractor.run_on_image(current_image)
            
            if args.filtering:
                print("---- EVALUATING WITH AD-HOC POST-PROCESSING FILTERING")
                predictions = filtering_process(predictions, per_image["concepts"], coco2synset, metadata)

            predictions = {
                'gt_boxes': predictions['pred_boxes'],
                'gt_classes': predictions['pred_classes'],
            }

            current_image = utils.convert_image_to_rgb(current_image, cfg.INPUT.FORMAT)
            visualizer = Visualizer(current_image, metadata=metadata, scale=scale)
            # target_fields = per_image["instances"].get_fields() 
            target_fields = predictions
            # print(per_image["instances"].get_fields())  #{'gt_boxes': Boxes(tensor([[720.5811, 601.0906, 803.6583, 672.0000]])), 'gt_classes': tensor([0])}
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            output(vis, str(per_image["image_id"]) + ".jpg")