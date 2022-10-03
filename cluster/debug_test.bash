#!/bin/bash

CONFIG_FILE=./configs/COCO/retinanet/retinanet_r50_fpn_COCO_concepts_test_add.yaml
OUTPUT_DIR="./results/debug/"

python train_net.py --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    --eval-only \
                    OUTPUT_DIR ${OUTPUT_DIR} \
                    SOLVER.IMS_PER_BATCH 16




