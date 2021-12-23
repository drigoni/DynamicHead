#!/usr/bin/env bash
# inputs
MODE=$1
VERSION=$2
CMD=$3
# default paths
PATH_DATASETS="$(pwd)/../../datasets"
PATH_FOLDER="$(pwd)"

if [[ $MODE == "build" ]]; then
  # build container
  docker build ./ -t $VERSION
  docker run -v $PATH_FOLDER/:/home/drigoni/repository/DynamicHead/ \
    --runtime=nvidia \
    $VERSION \
    python -m pip install -e /home/drigoni/repository/DynamicHead/
elif [[ $MODE == "exec" ]]; then
  # execute container
  docker run -it  \
    -v $PATH_FOLDER/:/home/drigoni/repository/DynamicHead/ \
    -v $PATH_FOLDER/datasets:/home/drigoni/repository/DynamicHead/datasets \
    -v $PATH_FOLDER/output:/home/drigoni/repository/DynamicHead/output \
    -v $PATH_FOLDER/demo:/home/drigoni/repository/DynamicHead/demo \
    -v $PATH_FOLDER/extracted_features:/home/drigoni/repository/DynamicHead/extracted_features \
    -v $PATH_FOLDER/.vector_cache:/home/drigoni/repository/DynamicHead/.vector_cache \
    -v $PATH_DATASETS/COCO/annotations:/home/drigoni/repository/DynamicHead/datasets/coco/annotations \
    -v $PATH_DATASETS/COCO/images/train2017:/home/drigoni/repository/DynamicHead/datasets/coco/train2017 \
    -v $PATH_DATASETS/COCO/images/val2017:/home/drigoni/repository/DynamicHead/datasets/coco/val2017 \
    -v $PATH_DATASETS/COCO/images/test2017:/home/drigoni/repository/DynamicHead/datasets/coco/test2017 \
    -v $PATH_FOLDER/datasets/flickr30k:/home/drigoni/repository/DynamicHead/datasets/flickr30k \
    -v $PATH_FOLDER/datasets/flickr30k/out_ewiser:/home/drigoni/repository/DynamicHead/datasets/flickr30k/out_ewiser \
    --runtime=nvidia \
    --ipc=host \
    $VERSION \
    $CMD
    # '{"mode":0, "dataset":"flickr30k", "suffix":"kl1n0.4", "prefetch_factor":10, "num_workers":30, "align_loss":"kl", "regression_loss":"reg", "restore": null, "loss_weight_reg":1.0, "align_loss_kl_threshold":0.4}'
else
  echo "To be implemented."
fi
