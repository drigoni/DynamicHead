#!/bin/bash

CONFIG_FILE=./configs/COCO/dh/dh_swint_fpn_COCO_concepts_train_add.yaml
MODEL_WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl"
OUTPUT_DIR="./results/debug/"

python train_net.py --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    MODEL.WEIGHTS ${MODEL_WEIGHTS}  \
                    DEEPSETS.EMB 'random' \
                    CONCEPT.APPLY_CONDITION False \
                    CONCEPT.APPLY_FILTER False \
                    OUTPUT_DIR ${OUTPUT_DIR} \
                    SOLVER.IMS_PER_BATCH 16




