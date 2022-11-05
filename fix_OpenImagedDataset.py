import json
import nltk
from nltk.corpus import wordnet as wn
from detectron2.data import detection_utils as utils
from tqdm import tqdm


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            print(dataset_dict)
    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def check_images(IMAGES, ANNOTATIONS):
    print("Loading annotations: ", ANNOTATIONS)
    with open(ANNOTATIONS, "r") as f:
        data = json.load(f)

    # print(data.keys())  # dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])
    # print(data['images'][0].keys()) # dict_keys(['id', 'file_name', 'neg_category_ids', 'pos_category_ids', 'width', 'height'])
    print("Checking... ")
    for img in tqdm(data['images']):
        image = utils.read_image(IMAGES + img["file_name"], format='BGR')
        check_image_size(img, image)


def fix_images(IMAGES, ANNOTATIONS):
    # fixing images values
    img_to_fix = ['2c4fb0d424f961fd', 'b55ff3bd0805c9de', '33f643e863732881', '1366cde3b480a15c',
                '2c180e4d18b6115f','847d5b0a9f45de6c', 'daad793337ca67ad', '0224c94fe59bcf46', '03a53ed6ab408b9f', '13c94e386c89bb54',
                '3095084b358d3f2d', '24e33d9be83efede', 'c538fa487b13429f', '52b1692ece39f48e', '17e017d9f38ada01', '1d5ef05c8da80e31', 
                '9e667cf88a27b4c1', '28b2e4ccdebad63d', '42c35de2aadd8cfc', '179013896a60264a', '0b61cfd27ff3b3bc', '7a41f1e085d6240e', 
                '000e4617274e83fe', '1af17f3d912e9aac', 'f84d7fc7f9cb9e7f', '42a30d8f8fba8b40', 'b319106f1df03a3f', 'e3c5bc6f6b004430']
    print("Loading annotations: ", ANNOTATIONS)
    with open(ANNOTATIONS, "r") as f:
        data = json.load(f)

    print("Checking... ")
    for img in tqdm(data['images']):
        if img['id'] in img_to_fix:
            image = utils.read_image(IMAGES + img["file_name"], format='BGR')
            height = image.shape[0]
            width = image.shape[1]
            img['height'] = height
            img['width'] = width
    NEW_ANNOTATIONS = ANNOTATIONS+'_fixSize'
    print("Saving: ", NEW_ANNOTATIONS)
    with open(NEW_ANNOTATIONS, "w") as f:
        json.dump(data, f)


SPLITS = ['test', 'val', 'train']
# SPLITS = ['train']
for split in SPLITS:
    IMAGES = "./datasets/OpenImagesDataset/{}/".format(split)
    # ANNOTATIONS = "./datasets/OpenImagesDataset/annotations/openimages_v4_{}_bbox.json".format(split)
    ANNOTATIONS = "./datasets/concept_OpenImagesDataset/annotations/openimages_v4_{}_bbox_subset_old.json".format(split)
    # ANNOTATIONS = "./datasets/concept_OpenImagesDataset/annotations/openimages_v4_{}_bbox_all.json".format(split)
    # ANNOTATIONS = "./datasets/concept_OpenImagesDataset/annotations/openimages_v4_{}_bbox_subset_old.json".format(split)
    # check_images(IMAGES, ANNOTATIONS)
    fix_images(IMAGES, ANNOTATIONS)

# valid
# {'id': '2c4fb0d424f961fd', 'file_name': '2c4fb0d424f961fd.jpg', 'neg_category_ids': [221, 434, 503, 552], 'pos_category_ids': [], 'width': 3648, 'height': 2736}
# {'id': 'b55ff3bd0805c9de', 'file_name': 'b55ff3bd0805c9de.jpg', 'neg_category_ids': [], 'pos_category_ids': [334, 456], 'width': 4608, 'height': 3456}
# {'id': '33f643e863732881', 'file_name': '33f643e863732881.jpg', 'neg_category_ids': [15, 267, 126, 263, 478, 484, 527, 561], 'pos_category_ids': [456], 'width': 6000, 'height': 4000}
# {'id': '1366cde3b480a15c', 'file_name': '1366cde3b480a15c.jpg', 'neg_category_ids': [160, 177, 221, 447], 'pos_category_ids': [69, 308, 433, 503, 573], 'width': 4320, 'height': 2432}

# test
# {'id': '2c180e4d18b6115f', 'file_name': '2c180e4d18b6115f.jpg', 'neg_category_ids': [292, 122], 'pos_category_ids': [91, 298, 561], 'width': 2560, 'height': 1920}
# {'id': '847d5b0a9f45de6c', 'file_name': '847d5b0a9f45de6c.jpg', 'neg_category_ids': [15, 148, 292, 409, 502, 568, 31, 83], 'pos_category_ids': [], 'width': 4000, 'height': 3000}
# {'id': 'daad793337ca67ad', 'file_name': 'daad793337ca67ad.jpg', 'neg_category_ids': [318, 408, 409, 552, 485], 'pos_category_ids': [], 'width': 4320, 'height': 3240}
# {'id': '0224c94fe59bcf46', 'file_name': '0224c94fe59bcf46.jpg', 'neg_category_ids': [298, 456], 'pos_category_ids': [334], 'width': 3648, 'height': 2736}
# {'id': '03a53ed6ab408b9f', 'file_name': '03a53ed6ab408b9f.jpg', 'neg_category_ids': [], 'pos_category_ids': [334, 391], 'width': 3568, 'height': 2368}
# {'id': '13c94e386c89bb54', 'file_name': '13c94e386c89bb54.jpg', 'neg_category_ids': [], 'pos_category_ids': [], 'width': 6000, 'height': 4000}
# {'id': 'e96d905ea321195e', 'file_name': 'e96d905ea321195e.jpg', 'neg_category_ids': [253, 573, 587], 'pos_category_ids': [301, 322, 332], 'width': 5472, 'height': 3648}

# train
# {'id': '3095084b358d3f2d', 'file_name': '3095084b358d3f2d.jpg', 'neg_category_ids': [], 'pos_category_ids': [193], 'width': 4240, 'height': 2832}
# {'id': '3ad7415a11ac1f5e', 'file_name': '3ad7415a11ac1f5e.jpg', 'neg_category_ids': [456, 51, 391], 'pos_category_ids': [69, 148, 177, 253, 298, 308, 334, 433, 503, 573], 'width': 6000, 'height': 4000}
# {'id': '24e33d9be83efede', 'file_name': '24e33d9be83efede.jpg', 'neg_category_ids': [121, 409, 528, 552], 'pos_category_ids': [69, 281, 433, 434, 466], 'width': 6016, 'height': 4016}
# {'id': 'c538fa487b13429f', 'file_name': 'c538fa487b13429f.jpg', 'neg_category_ids': [], 'pos_category_ids': [466], 'width': 4912, 'height': 2760}
# {'id': '52b1692ece39f48e', 'file_name': '52b1692ece39f48e.jpg', 'neg_category_ids': [295, 462], 'pos_category_ids': [103, 241, 334, 391, 404, 409, 485, 552, 571], 'width': 4912, 'height': 3264}
# {'id': '17e017d9f38ada01', 'file_name': '17e017d9f38ada01.jpg', 'neg_category_ids': [468, 572], 'pos_category_ids': [69, 103, 404, 409, 414, 571], 'width': 4496, 'height': 3000}
# {'id': '1d5ef05c8da80e31', 'file_name': '1d5ef05c8da80e31.jpg', 'neg_category_ids': [462], 'pos_category_ids': [69, 269, 447], 'width': 3648, 'height': 2736}
# {'id': '9e667cf88a27b4c1', 'file_name': '9e667cf88a27b4c1.jpg', 'neg_category_ids': [333, 434, 567], 'pos_category_ids': [69, 228, 433, 502], 'width': 4000, 'height': 3000}
# {'id': '28b2e4ccdebad63d', 'file_name': '28b2e4ccdebad63d.jpg', 'neg_category_ids': [51, 228, 334], 'pos_category_ids': [69, 308, 391, 434], 'width': 6000, 'height': 4000}
# {'id': '42c35de2aadd8cfc', 'file_name': '42c35de2aadd8cfc.jpg', 'neg_category_ids': [456], 'pos_category_ids': [203, 334], 'width': 4320, 'height': 2880}
# {'id': '179013896a60264a', 'file_name': '179013896a60264a.jpg', 'neg_category_ids': [38, 193, 334], 'pos_category_ids': [11], 'width': 4608, 'height': 3456}
# {'id': '0b61cfd27ff3b3bc', 'file_name': '0b61cfd27ff3b3bc.jpg', 'neg_category_ids': [69, 334], 'pos_category_ids': [91, 446, 561], 'width': 6000, 'height': 4000}
# {'id': '7a41f1e085d6240e', 'file_name': '7a41f1e085d6240e.jpg', 'neg_category_ids': [], 'pos_category_ids': [334, 456], 'width': 2448, 'height': 1624}
# {'id': '000e4617274e83fe', 'file_name': '000e4617274e83fe.jpg', 'neg_category_ids': [84, 87, 92, 160, 211, 213, 257, 366], 'pos_category_ids': [456, 485], 'width': 6016, 'height': 4000}
# {'id': '1af17f3d912e9aac', 'file_name': '1af17f3d912e9aac.jpg', 'neg_category_ids': [50], 'pos_category_ids': [571, 334, 391, 404], 'width': 3568, 'height': 2368}
# {'id': 'f84d7fc7f9cb9e7f', 'file_name': 'f84d7fc7f9cb9e7f.jpg', 'neg_category_ids': [], 'pos_category_ids': [219, 301], 'width': 5152, 'height': 3864}
# {'id': '42a30d8f8fba8b40', 'file_name': '42a30d8f8fba8b40.jpg', 'neg_category_ids': [51, 148, 221, 409, 552], 'pos_category_ids': [15, 69, 145, 177, 224, 228, 253, 292, 298, 308, 433, 502, 503, 568, 573], 'width': 4592, 'height': 3056}
# {'id': 'b319106f1df03a3f', 'file_name': 'b319106f1df03a3f.jpg', 'neg_category_ids': [308, 333, 502], 'pos_category_ids': [69], 'width': 6016, 'height': 4000}
# {'id': 'e3c5bc6f6b004430', 'file_name': 'e3c5bc6f6b004430.jpg', 'neg_category_ids': [334, 456, 466], 'pos_category_ids': [359], 'width': 4000, 'height': 3000}





# -- Process 3 terminated with the following error:
# Traceback (most recent call last):
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/multiprocessing/spawn.py", line 59, in _wrap
#     fn(i, *args)
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/engine/launch.py", line 126, in _distributed_worker
#     main_func(*args)
#   File "/ceph/hpc/scratch/user/eudavider/repository/DynamicHead/train_net.py", line 330, in main
#     return trainer.train()
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 484, in train
#     super().train(self.start_iter, self.max_iter)
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 149, in train
#     self.run_step()
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/engine/defaults.py", line 494, in run_step
#     self._trainer.run_step()
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/engine/train_loop.py", line 267, in run_step
#     data = next(self._data_loader_iter)
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/data/common.py", line 234, in __iter__
#     for d in self.dataset:
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 521, in __next__
#     data = self._next_data()
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1203, in _next_data
#     return self._process_data(data)
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 1229, in _process_data
#     data.reraise()
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/_utils.py", line 434, in reraise
#     raise exception
# detectron2.data.detection_utils.SizeMismatchError: Caught SizeMismatchError in DataLoader worker process 15.
# Original Traceback (most recent call last):
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
#     data = fetcher.fetch(index)
#   File "/ceph/hpc/home/eudavider/.conda/envs/dynamicHead/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 32, in fetch
#     data.append(next(self.dataset_iter))
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/data/common.py", line 201, in __iter__
#     yield self.dataset[idx]
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/data/common.py", line 90, in __getitem__
#     data = self._map_func(self._dataset[cur_idx])
#   File "/ceph/hpc/scratch/user/eudavider/repository/DynamicHead/extra/concept/concept_mapper.py", line 279, in __call__
#     utils.check_image_size(dataset_dict, image)
#   File "/ceph/hpc/home/eudavider/.local/lib/python3.9/site-packages/detectron2/data/detection_utils.py", line 196, in check_image_size
#     raise SizeMismatchError(
# detectron2.data.detection_utils.SizeMismatchError: Mismatched image shape for image datasets/OpenImagesDataset/train/000e4617274e83fe.jpg, got (4000, 6016), expect (6016, 4000). Please check the width/height in your annotation.
