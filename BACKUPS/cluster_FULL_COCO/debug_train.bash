#!/bin/bash

CONFIG_FILE=./configs/COCO/frcnn/r50/frcnn_r50_fpn_COCO_concepts_train_cat.yaml
MODEL_WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl"
OUTPUT_DIR="./results/debug/"

python train_net.py --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    MODEL.WEIGHTS ${MODEL_WEIGHTS}  \
                    DEEPSETS.EMB 'random' \
                    CONCEPT.APPLY_CONDITION True \
                    CONCEPT.APPLY_FILTER True \
                    OUTPUT_DIR ${OUTPUT_DIR} \
                    SOLVER.IMS_PER_BATCH 16  \
                    CONCEPT.DEPTH 3
                    # DATASETS.TRAIN '("coco_2017_tuning_val",)'




