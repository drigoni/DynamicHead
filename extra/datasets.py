import os
import contextlib
import io
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

# ==== Predefined splits for the new splits of data regarding concepts ===========
_PREDEFINED_CONCEPTS_SPLITS = {}
_PREDEFINED_CONCEPTS_SPLITS['coco'] = {
    "coco_2017_val_subset": ("coco/val2017",
                            "concept_coco/annotations/instances_val2017_subset.json"),
    "coco_2017_val_powerset": ("coco/val2017",
                                "concept_coco/annotations/instances_val2017_powerset.json"),
    "coco_2017_tuning_train": ("coco/train2017",
                                "tuning_coco/annotations/tuning_instances_train2017.json"),
    "coco_2017_tuning_val": ("coco/train2017",
                                "tuning_coco/annotations/tuning_instances_val2017.json"),
    "coco_2017_tuning_val_subset": ("coco/train2017",
                                    "concept_coco/annotations/tuning_instances_val2017_subset.json"),
    "coco_2017_tuning_val_powerset": ("coco/train2017",
                                        "concept_coco/annotations/tuning_instances_val2017_powerset.json"),
}

_PREDEFINED_CONCEPTS_SPLITS['vg'] = {
    "vg_train": ("visual_genome/images",
                "visual_genome/annotations/visual_genome_train.json"),
    "vg_val": ("visual_genome/images",
                "visual_genome/annotations/visual_genome_val.json"),
    "vg_test": ("visual_genome/images",
                "visual_genome/annotations/visual_genome_test.json"),
    "vg_val_subset": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_val_subset.json"),
    "vg_test_subset": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_test_subset.json"),
}

_PREDEFINED_CONCEPTS_SPLITS['oid'] = {
    "oid_v4_train": ("OpenImagesDataset/train/",
                    "OpenImagesDataset/annotations/openimages_v4_train_bbox.json"),
    "oid_v4_val": ("OpenImagesDataset/val/",
                    "OpenImagesDataset/annotations/openimages_v4_val_bbox.json"),
    "oid_v4_test": ("OpenImagesDataset/test/",
                    "OpenImagesDataset/annotations/openimages_v4_test_bbox.json"),
    "oid_v4_val_subset": ("OpenImagesDataset/val/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_val_bbox_subset.json"),
    "oid_v4_val_powerset": ("OpenImagesDataset/val/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_val_bbox_powerset.json"),
    "oid_v4_tes_subset": ("OpenImagesDataset/test/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_test_bbox_subset.json"),
}


# def _get_coco_instances_meta():
#     thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
#     thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
#     assert len(thing_ids) == 80, len(thing_ids)
#     # Mapping from the incontiguous COCO category id to an id in [0, 79]
#     thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
#     thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
#     ret = {
#         "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
#         "thing_classes": thing_classes,
#         "thing_colors": thing_colors,
#     }
#     return ret

def _get_vg_instances_meta():
    # This is for compatibility with COCO
    thing_dataset_id_to_contiguous_id = {i: i for i in range(0, 1600)} # VG annotations start from 0, not 1 as in COCO, and are in [0, 1599]. Background class is not included
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
    }
    return ret

def _get_oid_instances_meta():
    # This is for compatibility with COCO
    thing_dataset_id_to_contiguous_id = {i: i-1 for i in range(1, 602)}  # OID annotations start from 1 as COCO and are in [1, 601]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    elif dataset_name == 'vg':
        return _get_vg_instances_meta()
    elif dataset_name == 'oid':
        return _get_oid_instances_meta()
    else:
        raise KeyError("No built-in metadata for dataset {}".format(dataset_name))


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).
    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_datasets(root):
    for dataset_name in _PREDEFINED_CONCEPTS_SPLITS.keys():
        for key, (image_root, json_file) in _PREDEFINED_CONCEPTS_SPLITS[dataset_name].items():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file),
                os.path.join(root, image_root),
            )
        

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_datasets(_root)


def flatten_json(json):
    if type(json) == dict:
        for k, v in list(json.items()):
            if type(v) == dict:
                flatten_json(v)
                json.pop(k)
                for k2, v2 in v.items():
                    json[k+"."+k2] = v2


def unflatten_json(json):
    if type(json) == dict:
        for k in sorted(json.keys(), reverse=True):
            if "." in k:
                key_parts = k.split(".")
                json1 = json
                for i in range(0, len(key_parts)-1):
                    k1 = key_parts[i]
                    if k1 in json1:
                        json1 = json1[k1]
                        if type(json1) != dict:
                            conflicting_key = ".".join(key_parts[0:i+1])
                            raise Exception('Key "{}" conflicts with key "{}"'.format(
                                k, conflicting_key))
                    else:
                        json2 = dict()
                        json1[k1] = json2
                        json1 = json2
                if type(json1) == dict:
                    v = json.pop(k)
                    json1[key_parts[-1]] = v

