#!/usr/bin/env bash
# inputs
MODE=$1
VERSION=$2
GPU=$3
CMD=$4

# default paths
DATASETS_PATH="$(pwd)/../../datasets"
CURRENT_FOLDER="$(pwd)"
WANDB_KEY=06de2b089b5d98ee67dcf4fdffce3368e8bac2e4
USER=dkm
USER_ID=1003
USER_GROUP=dkm
USER_GROUP_ID=1003

if [[ $MODE == "build" ]]; then
  # build container
  docker build ./ -t $VERSION
  # here we use root permission instead of # -u
  docker run -v $CURRENT_FOLDER/:/home/drigoni/repository/DynamicHead/ \
    -u root\
    --runtime=nvidia \
    $VERSION \
    python -m pip install -e /home/drigoni/repository/DynamicHead/
elif [[ $MODE == "exec" ]]; then
  echo "Remove previous container: "
  docker container rm ${VERSION}-${GPU//,}
  # execute container
  echo "Execute container:"
  docker run \
    -u ${USER}:${USER_GROUP} \
    --env CUDA_VISIBLE_DEVICES=${GPU} \
    --env WANDB_API_KEY=${WANDB_KEY}\
    --name ${VERSION}-${GPU//,} \
    --runtime=nvidia \
    --ipc=host \
    -it  \
    -v ${CURRENT_FOLDER}/:/home/drigoni/repository/DynamicHead/ \
    -v ${CURRENT_FOLDER}/datasets:/home/drigoni/repository/DynamicHead/datasets \
    -v ${CURRENT_FOLDER}/output/:/home/drigoni/repository/DynamicHead/output \
    -v ${CURRENT_FOLDER}/demo:/home/drigoni/repository/DynamicHead/demo \
    -v ${CURRENT_FOLDER}/extracted_features:/home/drigoni/repository/DynamicHead/extracted_features \
    -v ${CURRENT_FOLDER}/.vector_cache:/home/drigoni/repository/DynamicHead/.vector_cache \
    -v ${CURRENT_FOLDER}/corpora:/root/nltk_data/corpora \
    -v ${CURRENT_FOLDER}/datasets/flickr30k:/home/drigoni/repository/DynamicHead/datasets/flickr30k \
    -v ${CURRENT_FOLDER}/datasets/flickr30k/out_ewiser:/home/drigoni/repository/DynamicHead/datasets/flickr30k/out_ewiser \
    -v ${DATASETS_PATH}/COCO/annotations:/home/drigoni/repository/DynamicHead/datasets/coco/annotations \
    -v ${DATASETS_PATH}/COCO/images/train2017:/home/drigoni/repository/DynamicHead/datasets/coco/train2017 \
    -v ${DATASETS_PATH}/COCO/images/val2017:/home/drigoni/repository/DynamicHead/datasets/coco/val2017 \
    -v ${DATASETS_PATH}/COCO/images/test2017:/home/drigoni/repository/DynamicHead/datasets/coco/test2017 \
    -v ${DATASETS_PATH}/VisualGenome/images/:/home/drigoni/repository/DynamicHead/datasets/visual_genome/images \
    -v ${DATASETS_PATH}/VisualGenome/annotations/:/home/drigoni/repository/DynamicHead/datasets/visual_genome/annotations \
    $VERSION \
    $CMD
    # '{"mode":0, "dataset":"flickr30k", "suffix":"kl1n0.4", "prefetch_factor":10, "num_workers":30, "align_loss":"kl", "regression_loss":"reg", "restore": null, "loss_weight_reg":1.0, "align_loss_kl_threshold":0.4}'
elif [[ $MODE == "interactive" ]]; then
  docker run -v $CURRENT_FOLDER/:/home/drigoni/repository/DynamicHead/ \
    -u root\
    --runtime=nvidia \
    -it \
    $VERSION \
    '/bin/bash'
else
  echo "To be implemented."
fi