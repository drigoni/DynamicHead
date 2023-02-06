#!/bin/bash

CONFIG_FILE=./configs/COCO/dh/r101/dh_r101_fpn_COCO_concepts_test_cat.yaml

python train_net.py \
                    --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    --num-machines 1 \
                    --eval-only \
                    DATASETS.TEST '("coco_2017_val_subset_old",)' \
                    CONCEPT.APPLY_CONDITION True \
                    EVALUATOR_TYPE default \



