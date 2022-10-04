#!/bin/bash

CONFIG_FILE=./configs/COCO/dh/dh_swint_fpn_COCO_test.yaml
OUTPUT_DIR="./results/debug/"

python train_net.py --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    --eval-only \
                    OUTPUT_DIR ${OUTPUT_DIR} \
                    SOLVER.IMS_PER_BATCH 16 \
                    EVALUATOR_TYPE 'GTFilter' # ['default', 'GTFilter']




