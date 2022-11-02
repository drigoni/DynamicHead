import os
import contextlib
import io
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from fvcore.common.timer import Timer
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

# ==== Predefined splits for the new splits of data regarding concepts ===========
# _PREDEFINED_CONCEPTS_SPLITS = {}
# _PREDEFINED_CONCEPTS_SPLITS['coco'] = {
#     "coco_2017_val_subset": ("coco/val2017",
#                             "concept_coco/annotations/instances_val2017_subset.json"),
#     "coco_2017_val_powerset": ("coco/val2017",
#                                 "concept_coco/annotations/instances_val2017_powerset.json"),
#     "coco_2017_tuning_train": ("coco/train2017",
#                                 "tuning_coco/annotations/tuning_instances_train2017.json"),
#     "coco_2017_tuning_val": ("coco/train2017",
#                                 "tuning_coco/annotations/tuning_instances_val2017.json"),
#     "coco_2017_tuning_val_subset": ("coco/train2017",
#                                     "concept_coco/annotations/tuning_instances_val2017_subset.json"),
#     "coco_2017_tuning_val_powerset": ("coco/train2017",
#                                         "concept_coco/annotations/tuning_instances_val2017_powerset.json"),
#     "coco_2017_tuning_val_subset_v2": ("coco/train2017",
#                                     "concept_coco_new/annotations/tuning_instances_val2017_subset.json"),
# }

# _PREDEFINED_CONCEPTS_SPLITS['vg'] = {
#     "vg_train": ("visual_genome/images",
#                 "visual_genome/annotations/visual_genome_train.json"),
#     "vg_val": ("visual_genome/images",
#                 "visual_genome/annotations/visual_genome_val.json"),
#     "vg_test": ("visual_genome/images",
#                 "visual_genome/annotations/visual_genome_test.json"),
#     "vg_val_subset": ("visual_genome/images",
#                         "concept_visual_genome/annotations/visual_genome_val_subset.json"),
#     "vg_test_subset": ("visual_genome/images",
#                         "concept_visual_genome/annotations/visual_genome_test_subset.json"),
# }

# _PREDEFINED_CONCEPTS_SPLITS['oid'] = {
#     "oid_v4_train": ("OpenImagesDataset/train/",
#                     "OpenImagesDataset/annotations/openimages_v4_train_bbox.json"),
#     "oid_v4_val": ("OpenImagesDataset/val/",
#                     "OpenImagesDataset/annotations/openimages_v4_val_bbox.json"),
#     "oid_v4_test": ("OpenImagesDataset/test/",
#                     "OpenImagesDataset/annotations/openimages_v4_test_bbox.json"),
#     "oid_v4_val_subset": ("OpenImagesDataset/val/",
#                             "concept_OpenImagesDataset/annotations/openimages_v4_val_bbox_subset.json"),
#     "oid_v4_val_powerset": ("OpenImagesDataset/val/",
#                             "concept_OpenImagesDataset/annotations/openimages_v4_val_bbox_powerset.json"),
#     "oid_v4_tes_subset": ("OpenImagesDataset/test/",
#                             "concept_OpenImagesDataset/annotations/openimages_v4_test_bbox_subset.json"),
# }

_PREDEFINED_CONCEPTS_SPLITS = {}
_PREDEFINED_CONCEPTS_SPLITS['coco'] = {
    # already registered
    # "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    # "coco_2017_test": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    # NOTE:  "coco_2017_val" is our "coco_2017_test" test set
    "coco_2017_train_all": ("coco/train2017",
                                    "concept_coco/annotations/instances_train2017_all.json"),
    "coco_2017_train_subset_old": ("coco/train2017",
                                    "concept_coco/annotations/instances_train2017_subset_old.json"),     
    "coco_2017_val_all": ("coco/val2017",
                                    "concept_coco/annotations/instances_val2017_all.json"),
    "coco_2017_val_subset_old": ("coco/val2017",
                                    "concept_coco/annotations/instances_val2017_subset_old.json"),     
    # tuning version                  
    "coco_2017_tuning_train": ("coco/train2017",
                                "tuning_coco/annotations/tuning_instances_train2017.json"),
    "coco_2017_tuning_val": ("coco/train2017",
                                "tuning_coco/annotations/tuning_instances_val2017.json"),
    "coco_2017_tuning_train_all": ("coco/train2017",
                                    "concept_tuning_coco/annotations/tuning_instances_train2017_all.json"),
    "coco_2017_tuning_train_subset_old": ("coco/train2017",
                                    "concept_tuning_coco/annotations/tuning_instances_train2017_subset_old.json"),
    "coco_2017_tuning_val_all": ("coco/train2017",
                                    "concept_tuning_coco/annotations/tuning_instances_val2017_all.json"),
    "coco_2017_tuning_val_subset_old": ("coco/train2017",
                                    "concept_tuning_coco/annotations/tuning_instances_val2017_subset_old.json"),
}

_PREDEFINED_CONCEPTS_SPLITS['vg'] = {
    "vg_train": ("visual_genome/images",
                "visual_genome/annotations/visual_genome_train.json"),
    "vg_val": ("visual_genome/images",
                "visual_genome/annotations/visual_genome_val.json"),
    "vg_test": ("visual_genome/images",
                "visual_genome/annotations/visual_genome_test.json"),
    # concept
    "vg_train_subset_old": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_train_subset_old.json"),
    "vg_train_all": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_train_all.json"),
    "vg_val_subset_old": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_val_subset_old.json"),
    "vg_val_all": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_val_all.json"),
    "vg_test_subset_old": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_test_subset_old.json"),
    "vg_test_all": ("visual_genome/images",
                        "concept_visual_genome/annotations/visual_genome_test_all.json"),
}

_PREDEFINED_CONCEPTS_SPLITS['oid'] = {
    "oid_v4_train": ("OpenImagesDataset/train/",
                    "OpenImagesDataset/annotations/openimages_v4_train_bbox.json"),
    "oid_v4_val": ("OpenImagesDataset/val/",
                    "OpenImagesDataset/annotations/openimages_v4_val_bbox.json"),
    "oid_v4_test": ("OpenImagesDataset/test/",
                    "OpenImagesDataset/annotations/openimages_v4_test_bbox.json"),
    "oid_v4_train_subset_old": ("OpenImagesDataset/train/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_train_bbox_subset_old.json"),
    "oid_v4_train_all": ("OpenImagesDataset/train/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_train_bbox_all.json"),
    "oid_v4_val_subset_old": ("OpenImagesDataset/val/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_val_bbox_subset_old.json"),
    "oid_v4_val_all": ("OpenImagesDataset/val/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_val_bbox_all.json"),
    "oid_v4_tes_subset_old": ("OpenImagesDataset/test/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_test_bbox_subset_old.json"),
    "oid_v4_tes_all": ("OpenImagesDataset/test/",
                            "concept_OpenImagesDataset/annotations/openimages_v4_test_bbox_all.json"),
}


def load_coco_with_concepts_json(json_file,
                                   image_root,
                                   dataset_name=None,
                                   extra_annotation_keys=None):
    """
    Extend load_coco_json() with additional support for concepts
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        # get concepts
        record["concepts"] = img_dict["concepts"]
        # concepts = anno.get("concepts", None)
        #     if concepts:  # list[int]
        #         obj["concepts"] = concepts
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:
                if not isinstance(segm, dict):
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


# ==========================================================
# DATASET REGISTRATIONS
# ==========================================================
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
    if 'subset' in name or 'all' in name:
        loader_function = load_coco_with_concepts_json
    else:
        loader_function = load_coco_json
    DatasetCatalog.register(name, lambda: loader_function(json_file, image_root, name))

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





# ==========================================================
# UTILITIES
# ==========================================================
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

