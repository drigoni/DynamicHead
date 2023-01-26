#!/usr/bin/env python
"""
Created on 16/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed for creating the evaluation dataset.
"""
# import nltk
import os
import sys
import json
import argparse
import json
import random
import nltk
from collections import defaultdict
from itertools import chain, combinations
import distutils
import random
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))


# NOTE: this class is not anymore used. It add a new key in the dataset dictionary.
class MakeEvaluationDataset:
    def __init__(self, args, dataset_name="tuning_coco"):
        self.dataset_name = dataset_name
        self.n_valid_examples = args.n_valid_examples
        self.coco_dataset = args.coco_dataset

        # params
        self.coco_dataset = args.coco_dataset

    def generate_output_file_name(self):
        output_train_file = "./datasets/{}/annotations/tuning_instances_train2017.json".format(self.dataset_name)
        output_val_file = "./datasets/{}/annotations/tuning_instances_val2017.json".format(self.dataset_name)
        return output_train_file, output_val_file

    @staticmethod
    def save_dataset(output_file, annotations):
        # save concept_coco annotation file
        with open(output_file, 'w') as file:
            json.dump(annotations, file)
        print("Dataset saved in {} .".format(output_file))

    @staticmethod
    def load_dataset(annotations_file):
        print("Loading coco annotations in {} .".format(annotations_file))
        with open(annotations_file, 'r') as file:
            annotations = json.load(file)
        return annotations

    @staticmethod
    def update_dataset_info(annotations):
        annotations['info']['description'] = "COCO 2017 Dataset used for experiments."
        annotations['info']['contributor'] = "Davide Rigoni"
        annotations['info']['date_created'] = "2022/10/25"

    def generate_train_valid_sets(self, original_set):
        # to update:
        # "images": [...],
        # "annotations": [...],
        # "segment_info": [...] <-- Only in Panoptic annotations

        image_to_annotations = defaultdict(list)
        for i, ann_dict in enumerate(original_set['annotations']):
            image_id = ann_dict['image_id']
            image_to_annotations[image_id].append(ann_dict)

        image_list = dict()
        for i, img_dict in enumerate(original_set['images']):
            image_id = img_dict['id']
            image_list[image_id] = img_dict
        n_images = len(image_list.keys())

        # TODO: missing segmentation. But we do not need it.

        # check number of examples in the new validation
        assert n_images > self.n_valid_examples, print("Not enough examples in the original dataset to build new datasets.")
        
        # make new train and valid sets
        train_set = {}
        train_set['info'] = copy.deepcopy(original_set['info'])
        train_set['licenses'] = copy.deepcopy(original_set['licenses'])
        train_set['categories'] = copy.deepcopy(original_set['categories'])
        train_set['images'] = []
        train_set['annotations'] = []
        valid_set = {}
        valid_set['info'] = copy.deepcopy(original_set['info'])
        valid_set['licenses'] = copy.deepcopy(original_set['licenses'])
        valid_set['categories'] = copy.deepcopy(original_set['categories'])
        valid_set['images'] = []
        valid_set['annotations'] = []

        # select random indeces
        new_val_indexes = random.sample(range(0, n_images), self.n_valid_examples)
        for i, image_id in enumerate(image_list.keys()):
            # select set
            if i in new_val_indexes:
                current_set = valid_set
            else:
                current_set = train_set

            current_set['images'].append(image_list[image_id])
            current_set['annotations'].extend(image_to_annotations[image_id])
        
        print("Number of images in the new train set: ", len(train_set['images']))
        print("Number of annotations in the new train set: ", len(train_set['annotations']))
        print("Number of images in the new valid set: ", len(valid_set['images']))
        print("Number of annotations in the new valid set: ", len(valid_set['annotations']))

        return train_set, valid_set
    
    def create_evaluation_dataset(self):
        # read coco annotation file
        annotations = MakeEvaluationDataset.load_dataset(self.coco_dataset)

        # dictionary
        # "info": {...},
        # "licenses": [...],
        # "images": [...],
        # "annotations": [...],
        # "categories": [...], <-- Not in Captions annotations
        # "segment_info": [...] <-- Only in Panoptic annotations

        # write new dataset information
        print("Processing. ")
        # update information about the dataset
        MakeEvaluationDataset.update_dataset_info(annotations)

        # generate external file
        train_set, valid_set = self.generate_train_valid_sets(annotations)

        # save new dataset
        output_train_file, output_val_file = self.generate_output_file_name()
        MakeEvaluationDataset.save_dataset(output_train_file, train_set)
        MakeEvaluationDataset.save_dataset(output_val_file, valid_set)
        print("Dataset {} processed. ".format(self.coco_dataset))    



def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Root folder.',
                        default='{}/'.format(current_dir),
                        type=str)
    parser.add_argument('--coco_dataset', dest='coco_dataset',
                        help='COCO dataset training file.',
                        default="./datasets/coco/annotations/instances_train2017.json",
                        type=str)
    parser.add_argument('--n_valid_examples', dest='n_valid_examples',
                        help='Number of examples to consider in the new validation set.',
                        default=5000,
                        type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    maker = MakeEvaluationDataset(args)
    maker.create_evaluation_dataset()