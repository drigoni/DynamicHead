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
from detectron2.data.build import get_detection_dataset_dicts
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
from extra import ConceptMapper
from extra import add_concept_config, add_extra_config, ConceptFinder, inference_filtering_process
from train_net import Trainer

nltk.download('wordnet')
nltk.download('omw-1.4')



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

    def __init__(self, cfg, args=None):
        self.cfg = cfg.clone()  # cfg can be modified by model
        # self.model = build_model(self.cfg)
        self.model = Trainer.build_model(cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(cfg.MODEL.WEIGHTS)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        # NOTE: this is not needed because it is done in the mapper function of the loading dataset.
        # self.aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.cpu_device = torch.device("cpu")
        self.filtering = args is not None and args.filtering is True
        if self.filtering:
            print("---- EVALUATING WITH AD-HOC POST-PROCESSING FILTERING")
            concept_finder = ConceptFinder(cfg.CONCEPT.FILE, depth=cfg.CONCEPT.DEPTH, unique=cfg.CONCEPT.UNIQUE, only_name=cfg.CONCEPT.ONLY_NAME)
            self.coco2synset = concept_finder.coco2synset

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
        # if cfg.MODEL.META_ARCHITECTURE in ["CATSS", "ConceptGeneralizedRCNN", "ConceptRetinaNet"] and cfg.CONCEPT.APPLY_CONDITION:
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            # concepts pre-processing. NOTE: add ['entity.n.01']
            if cfg.MODEL.META_ARCHITECTURE in ["CATSS", "ConceptGeneralizedRCNN", "ConceptRetinaNet"]:
                if cfg.CONCEPT.APPLY_CONDITION and concepts is not None:
                    print("Using concepts: {}. ".format(concepts))
                elif cfg.CONCEPT.APPLY_CONDITION and concepts is None:
                    print("Error. Concept not available in input, but should be used. ")
                    exit(1)
                elif not cfg.CONCEPT.APPLY_CONDITION:
                    print("Concept available in input, but not used. ")
                    concepts = ['entity.n.01']
            else:
                if cfg.CONCEPT.APPLY_CONDITION and concepts is not None:
                    print("Error. Concepts available, and should be used. However, the architecture does not use them. ")
                    exit(1)

            # make predictions
            inputs = {"image": image, "height": height, "width": width, 'concepts': concepts}
            predictions = self.model([inputs])[0]
            
            # change format
            instances = predictions["instances"].to(self.cpu_device)
            results = dict()
            for k, v in instances.get_fields().items():
                if isinstance(v, Boxes):
                    boxes_list = v.tensor
                    results[k] = boxes_list.tolist()
                else:
                    results[k] = v.tolist()
            assert len(results['features']) ==  len(results['pred_boxes']) == len(results['probs']), 'Error in the results.'

            # filter results
            if self.filtering:
                results = inference_filtering_process(results, concepts, self.coco2synset, self.metadata)
            return results


class ProposalExtractor(object):
    def __init__(self, cfg, args=None):
        """
        Args:
            cfg (CfgNode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
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
        predictions = self.predictor(image, concepts)
        return predictions


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
    cfg.MODEL.ATSS.NMS_TH = args.nms_th                         # default: 0.6
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_th           # default: 0.5
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = args.nms_th           # default: 0.5

    cfg.TEST.DETECTIONS_PER_IMAGE = args.detection_per_image    # default: 100
    
    cfg.MODEL.ATSS.INFERENCE_TH = args.inference_th             # default: 0.05
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.inference_th   # default: 0.05
    # cfg.MODEL.ATSS.PRE_NMS_TOP_N = args.pre_nms_top_n               # default: 1000
    # cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = args.pre_nms_top_n   # default: 1000

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("--filtering", action="store_true", help="apply postprocessing filtering")
    parser.add_argument("--inference_th", default=0.05, type=float, help="Minimum score for instance predictions to be shown")
    parser.add_argument("--nms_th", default=0.6, type=float, help="cfg.MODEL.ATSS.NMS_TH")
    parser.add_argument("--detection_per_image", default=100, type=int, help="cfg.TEST.DETECTIONS_PER_IMAGE.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    # parse inputs
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    # make output folder
    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)

    # load dataset
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TEST,
        filter_empty=True
    )
    
    concept_finder = ConceptFinder(cfg.CONCEPT.FILE, depth=cfg.CONCEPT.DEPTH, unique=cfg.CONCEPT.UNIQUE, only_name=cfg.CONCEPT.ONLY_NAME)
    coco2synset = concept_finder.coco2synset
    mapper = ConceptMapper(cfg, False, coco2synset=coco2synset)
    test_data_loader = build_detection_test_loader(dataset, mapper=mapper)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    # load model
    extractor = ProposalExtractor(cfg, args)

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
    for batch in test_data_loader:
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            # current_image = read_image(per_image["file_name"], format="RGB")
            current_image = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()

            # check wether the concepts are used or not.
            current_concepts = per_image.get('concepts') 
            predictions = extractor.run_on_image(current_image, current_concepts)

            target_fields = dict()
            for k, v in per_image["instances"].get_fields().items():
                if isinstance(v, Boxes):
                    boxes_list = v.tensor
                    target_fields[k] = boxes_list.tolist()
                else:
                    target_fields[k] = v.tolist()
            data_to_plot = {
                'boxes': predictions['pred_boxes'] + target_fields["gt_boxes"],
                'classes': (tmp_cls:=predictions['pred_classes'] + target_fields["gt_classes"]),
                'labels': [metadata.thing_classes[i] for i in tmp_cls],
                'colors': ['blue' for i in predictions['pred_boxes']] + ['red' for i in target_fields['gt_boxes']],
            }

            current_image = utils.convert_image_to_rgb(current_image, cfg.INPUT.FORMAT)
            visualizer = Visualizer(current_image, metadata=metadata, scale=scale)
            # print(per_image["instances"].get_fields())  #{'gt_boxes': Boxes(tensor([[720.5811, 601.0906, 803.6583, 672.0000]])), 'gt_classes': tensor([0])}
            vis = visualizer.overlay_instances(
                labels=data_to_plot['labels'],
                boxes=data_to_plot['boxes'],
                masks=None,
                keypoints=None,
                assigned_colors=data_to_plot['colors'],
            )
            output(vis, str(per_image["image_id"]) + ".jpg")