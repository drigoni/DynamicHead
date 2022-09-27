#!/bin/bash

CONFIG_FILE=./configs/dh_COCO_concepts_train_zeros.yaml
MODEL_WEIGHTS=./pretrained/dyhead_swint_atss_fpn_2x_ms.pth
NUM_GPUS=1
DEEPSETS_EMB='random'
CONCEPT_APPLY_CONDITION=False
CONCEPT_APPLY_FILTER=False
OUTPUT_DIR="./results/debug/"

python train_net.py --config  ${CONFIG_FILE} \
                    --num-gpus ${NUM_GPUS} \
                    MODEL.WEIGHTS ${MODEL_WEIGHTS}  \
                    DEEPSETS.EMB ${DEEPSETS_EMB} \
                    CONCEPT.APPLY_CONDITION ${CONCEPT_APPLY_CONDITION} \
                    CONCEPT.APPLY_FILTER ${CONCEPT_APPLY_FILTER} \
                    OUTPUT_DIR ${OUTPUT_DIR} \
                    SOLVER.IMS_PER_BATCH 1
