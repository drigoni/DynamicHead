#!/bin/bash

CONFIG_FILE=./configs/VG/retinanet/r101/retinanet_r101_fpn_VG_concepts_test_cat.yaml

python train_net.py \
                    --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    --num-machines 1 \
                    --eval-only \
                    DATASETS.TEST '("vg_test_subset_old",)' \
                    CONCEPT.APPLY_CONDITION True \
                    EVALUATOR_TYPE default \




