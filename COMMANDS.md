# COMMANDS TO USE


## EWISER
```
python bin/annotate_drigoni_COCO_captions.py --input ./data/COCO/ --checkpoint ./ewiser.semcor+wngt.pt --output_dir ./out_COCO_captions_train/
```


## Make Concept Dataset
```
python make_concept_dataset.py  --coco_dataset ./datasets/coco/annotations/instances_val2017.json \
                                --level 10 \
                                --coco2concepts ./concept/coco_to_synset.json \
                                --unique true \
                                --subset true
```


## Training
```
CONFIG_FILE=./configs/dh_COCO_concepts_train_add.yaml
srun python train_net.py \
                    --config  ${CONFIG_FILE} \
                    --num-gpus ${NUM_GPUS} \
                    --num-machines ${NUM_MACHINES} \
                    --dist-url ${SLURM_MASTER_URL} \
                    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
                    SOLVER.MAX_ITER ${MAX_ITER} \
```


## Testing
```
python train_net.py --config configs/drigoni_dyhead_swint_catss_fpn_2x_ms_COCO_concepts.yaml --num-gpus 1 --eval-only  MODEL.WEIGHTS ./output_concepts_master_bighead/model_final.pth DEEPSETS.EMB 'random' CONCEPT.APPLY_CONDITION_FROM_FILE False CONCEPT.APPLY_CONDITION True CONCEPT.ACTIVATE_CONCEPT_GENERATOR False
```


## Testing with external concepts
```
python train_net.py --config configs/drigoni_dyhead_swint_catss_fpn_2x_ms_COCO_concepts.yaml --num-gpus 1 --eval-only  MODEL.WEIGHTS ./output_concepts_master_bighead/model_final.pth DEEPSETS.EMB 'random' CONCEPT.APPLY_CONDITION_FROM_FILE True
```


## Extract Features
```
python extract_features.py --config configs/drigoni_dyhead_swint_catss_fpn_2x_ms_VG_concepts.yaml --images_folder ./datasets/flickr30k/flickr30k_images/ --concepts ./datasets/out_ewiser/*.txt.json --output ./extracted_features/extracted_features_05_100_VG_concepts_sentence --parallel True --inference_th 0.05 --pre_nms_top_n 10000 --nms_th 0.6 --detection_per_image 100 --opts MODEL.WEIGHTS ./model_final_VG_concepts_sentence.pth
```


## Check External Concepts
```
python check_external_concepts.py --folder ./datasets/ewiser_concepts_COCO_valid/ --file ./datasets/coco/annotations/instances_val2017.json
python check_external_concepts.py --folder ./datasets/ewiser_concepts_COCO_valid/ --coco2concepts ./concept/coco_to_synset.json --level 10
```