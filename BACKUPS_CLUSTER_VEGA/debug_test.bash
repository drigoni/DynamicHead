#!/bin/bash

CONFIG_FILE=./configs/COCO/dh/r50/dh_r50_fpn_COCO_concepts_test_cat.yaml
DATSETS=('("coco_2017_val_all",)' '("coco_2017_val_all",)' '("coco_2017_val_all",)' '("coco_2017_val_all",)' \
         '("coco_2017_val_subset_old",)' '("coco_2017_val_subset_old",)' '("coco_2017_val_subset_old",)' )
APPLY_CONDITION=(True False True False \
                True False True False)
EVALUATOR_TYPE=('default' 'default' 'postProcessing' 'postProcessing' \
                'default' 'default' 'postProcessing' 'postProcessing')
python train_net.py \
                    --config  ${CONFIG_FILE} \
                    --num-gpus 1 \
                    --num-machines 1 \
                    --eval-only \
                    DATASETS.TEST '("coco_2017_val_subset_old",)' \
                    CONCEPT.APPLY_CONDITION True \
                    EVALUATOR_TYPE default \




