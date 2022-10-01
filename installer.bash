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
