# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
import argparse
import os
import sys
import torch
import tqdm
import cv2
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from utils import mkdir, save_features
from extract_utils import get_image_blob
from bua.caffe import add_bottom_up_attention_config
from bua.caffe.modeling.layers.nms import nms
from bua.d2 import add_attribute_config
import os
#!/usr/bin/env python
"""
Created on 20/12/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: 
"""
import atexit
import bisect
import multiprocessing as mp
import cv2
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

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import default_setup
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures.boxes import Boxes

from dyhead import add_dyhead_config
from extra import add_extra_config
from extra import add_concept_config
from train_net import Trainer

# constants
WINDOW_NAME = "COCO detections"


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data, concepts = task
                result = predictor(data, concepts)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image, concepts):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image, concepts))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res, concepts = self.result_queue.get()
            if idx == self.get_idx:
                return res, concepts
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, (res, concepts))

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image, concepts):
        self.put(image, concepts)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


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
            cfg.DATASETS.TEST.
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
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(cfg.MODEL.WEIGHTS)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, concepts):
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
    def __init__(self, cfg, parallel=False):
        """
        Args:
            cfg (CfgNode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        self.parallel = parallel
        if self.parallel == True:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, concepts):
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


def extract_flickr30k_concepts(ewiser_path):
    # reading data from file
    with open(ewiser_path, 'r') as f:
        ewiser_data = json.load(f)
    # extracting synsets
    all_synsets = []
    for sentence in ewiser_data:
        ewiser = sentence['ewiser']
        for part in ewiser:
            n = part['n_synsets']
            offsets = part['offsets']
            synsets = part['synsets']
            # select just one synset among the top10 and filter them to be just noun
            if n > 0:
                synsets_filtered = [s for s in synsets if '.n.' in s]
                if len(synsets_filtered) > 0:
                    best_synset = synsets_filtered[0]
                    all_synsets.append(best_synset)
    # check if there are at least one concept
    if len(all_synsets) == 0:
        print("No noun synset for {}.".format(ewiser_path))
    all_synsets_unique = list(set(all_synsets))
    return all_synsets_unique


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    add_concept_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ATSS.INFERENCE_TH = args.inference_th
    cfg.MODEL.ATSS.PRE_NMS_TOP_N = args.pre_nms_top_n
    cfg.MODEL.ATSS.NMS_TH = args.nms_th
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detection_per_image
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for Concept-Conditioned Models")
    parser.add_argument(
        "--config",
        default='/configs/COCO/retinanet/r50/retinanet_r50_fpn_COCO_concepts_test_cat.yaml',
        help="Path to config file",
    )
    parser.add_argument(
        "--parallel",
        help="=True if the GPUs are used",
        default=lambda x: True if x.lower() == 'true' else False,
    )
    parser.add_argument(
        "--images_folder",
        help="Images folder. "
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        help="A list of concepts file . "
    )
    parser.add_argument(
        "--output",
        default='./demo/',
        help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--inference_th",
        type=float,
        default=0.05,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--pre_nms_top_n",
        type=int,
        default=1000,
        help="cfg.MODEL.ATSS.PRE_NMS_TOP_N.",
    )
    parser.add_argument(
        "--nms_th",
        type=float,
        default=0.6,
        help="cfg.MODEL.ATSS.NMS_TH",
    )
    parser.add_argument(
        "--detection_per_image",
        type=int,
        default=30,
        help="cfg.TEST.DETECTIONS_PER_IMAGE.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    extractor = ProposalExtractor(cfg, args.parallel)

    n_proposals = []
    n_concepts = []

    images_folder = args.images_folder
    concepts_folder = os.path.dirname(args.concepts[0])
    for concept_path in tqdm.tqdm(args.concepts, disable=not args.output):
        concept_filename = os.path.basename(concept_path)
        image_filename = '{}.jpg'.format(concept_filename[:-9])  # remove ".txt.json" and add ".jpg"
        image_path = os.path.join(images_folder, image_filename)

        # get concepts
        current_concepts = extract_flickr30k_concepts(concept_path)

        # use PIL, to be consistent with evaluation
        img = read_image(image_path, format="BGR")
        start_time = time.time()
        predictions = extractor.run_on_image(img, current_concepts)
        logger.info(
            "Done {} in {:.2f}s with {} proposals and {} concepts".format(
                image_path,
                time.time() - start_time,
                len(predictions['pred_boxes']),
                len(current_concepts)
            )
        )

        # update statistics
        n_proposals.append(len(predictions['pred_boxes']))
        n_concepts.append(len(current_concepts))

        # save predictions
        out_filename = os.path.join(args.output, os.path.basename(image_path))
        out_filename = '{}.json'.format(out_filename[:-4])
        # logger.info(out_filename)
        # print('Features: ', len(predictions['features']), len(predictions['features'][0]))
        with open(out_filename, 'w') as outfile:
            json.dump(predictions, outfile)
