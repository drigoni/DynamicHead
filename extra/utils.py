import os
import contextlib
import io
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances


# ==== Predefined splits for the new splits of data regarding concepts ===========
_PREDEFINED_CONCEPTS_SPLITS_COCO = {
    "coco_2017_val_subset": ("coco/val2017",
                            "concept_coco/annotations/instances_val2017_subset.json"),
    "coco_2017_val_powerset": ("coco/val2017",
                                "concept_coco/annotations/instances_val2017_powerset.json"),
    "coco_2017_test_subset": ("coco/test2017",
                            "concept_coco/annotations/instances_test2017_subset.json"),
    "coco_2017_test_powerset": ("coco/test2017",
                                "concept_coco/annotations/instances_test2017_powerset.json"),
}


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


# def register_coco_instances_with_attributes(name, metadata, json_file, image_root):
#     DatasetCatalog.register(name, lambda: load_coco_with_attributes_json(json_file,
#                                                                          image_root,
#                                                                          name))
#     MetadataCatalog.get(name).set(
#         json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
#     )


def register_all_coco_datasets(root):
    for key, (image_root, json_file) in _PREDEFINED_CONCEPTS_SPLITS_COCO.items():
        register_coco_instances(
            key,
            {}, # no meta data
            os.path.join(root, json_file),
            os.path.join(root, image_root),
        )
        

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_datasets(_root)
