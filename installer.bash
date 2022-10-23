#!/bin/bash
# inputs
MODE=$1

if [[ $MODE == "install" ]]; then
  # conda create -n dynamicHead pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
  # conda create -n pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 torchtext -c pytorch
  # those work at some extent
  # conda create -n dynamicHead pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.6 torchtext -c pytorch -c conda-forge
  conda create -n dynamicHead pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 torchtext=0.11.0 -c pytorch -c conda-forge
  
  conda activate dynamicHead
  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
  python -m pip install -e /ceph/hpc/home/eudavider/repository/DynamicHead

  pip install nltk
  pip install numpy
  pip install setuptools==59.5.0  # https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version
  pip install wandb

  # due to a bug
  # pip uninstall torch
  # pip uninstall torchvision
  # pip install torch==1.10.0+cu111 torchvision -f https://download.pytorch.org/whl/torch_stable.html
  # pip install torchtext==0.11.0 # use this dependency because there is a problem with others.

elif [[ $MODE == "uninstall" ]]; then
  conda deactivate
  conda env remove -n dynamicHead
else
  echo "To be implemented."
fi





# convert OpenImagesDataset .csv files to COCO .json annotations: https://github.com/bethgelab/openimages2coco
# conda create -n tmp python=2, numpy, cython
# conda activate tmp
# pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
# git clone https://github.com/bethgelab/openimages2coco.git
# cd openimages2coco
# small bug to fix: substitute line 95 of convert_annotations.py with:
#    if image_size_sourcefile is not None:
#        original_image_sizes = utils.csvread(os.path.join('data/', image_size_sourcefile))
#    else:
#        original_image_sizes = None
# python3 convert_annotations.py -p /ceph/hpc/home/eudavider/datasets/OID_dataset/ --task bbox --subsets train --version v4


# make concept dataset
# python make_concept_dataset.py  --coco_dataset ./datasets/OpenImagesDataset/annotations/openimages_v4_val_bbox.json 
#                                 --level 2 --unique True --subset True --dataset_name concept_OpenImagesDataset
#                                 --coco2concepts ./concept/oid_to_synset.json 
# python make_concept_dataset.py  --coco_dataset ./datasets/coco/annotations/instances_val2017.json \
#                                  --level 2 --unique True --subset True
#
#

