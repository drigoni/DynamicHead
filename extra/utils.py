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
    "coco_2017_tuning_train": ("coco/train2017",
                                "tuning_coco/annotations/tuning_instances_train2017.json"),
    "coco_2017_tuning_val": ("coco/train2017",
                                "tuning_coco/annotations/tuning_instances_val2017.json"),
    "coco_2017_tuning_val_subset": ("coco/train2017",
                                    "concept_coco/annotations/tuning_instances_val2017_subset.json"),
    "coco_2017_tuning_val_powerset": ("coco/train2017",
                                        "concept_coco/annotations/tuning_instances_val2017_powerset.json"),

    "vg_train": ("visual_genome/images",
                "visual_genomes/annotations/visual_genome_train.json"),
    "vg_val": ("visual_genome/images",
                "visual_genomes/annotations/visual_genome_val.json"),
    "vg_test": ("visual_genome/images",
                "visual_genomes/annotations/visual_genome_test.json"),
    "vg_val_subset": ("visual_genome/images",
                        "concept_visual_genomes/annotations/visual_genome_val_subset.json"),
    "vg_test_subset": ("visual_genome/images",
                        "concept_visual_genomes/annotations/visual_genome_test_subset.json"),

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
