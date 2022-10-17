#!/bin/bash

CONFIG_FILE=./configs/COCO/retinanet/r50/retinanet_r50_fpn_COCO_concepts_train_cat.yaml
# CONFIG_FILE=./configs/COCO/frcnn/r50/frcnn_r50_fpn_COCO_concepts_train_cat.yaml
# CONFIG_FILE=./configs/OID/retinanet/r50/retinanet_r50_fpn_OID_concepts_train_cat.yaml
# CONFIG_FILE=./configs/VG/retinanet/r50/retinanet_drigoni_r50_fpn_VG_train.yaml
MODEL_WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl"
OUTPUT_DIR="./results/debug/"

python train_net.py --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    MODEL.WEIGHTS ${MODEL_WEIGHTS}  \
                    DEEPSETS.EMB 'random' \
                    CONCEPT.APPLY_CONDITION True \
                    CONCEPT.APPLY_FILTER True \
                    OUTPUT_DIR ${OUTPUT_DIR} \
                    SOLVER.IMS_PER_BATCH 1  \




