#!/usr/bin/env python
"""
Created on 16/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed for creating the new concept dataset.
"""
# import nltk
import os
import json
import argparse
import json
import random
import nltk
from collections import defaultdict
from itertools import chain, combinations
import distutils
from nltk.corpus import wordnet as wn
import copy
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from extra import ConceptFinder

current_dir = os.path.dirname(os.path.abspath(__file__))

nltk.download('omw-1.4')


# NOTE: this class is not anymore used. It add a new key in the dataset dictionary.
class MakeConceptDataset:
    def __init__(self, args):
        # download package
        nltk.download('wordnet')
        self.dataset_name = args.dataset_name

        # params
        self.coco_dataset = args.coco_dataset
        self.level = args.level
        self.unique = args.unique
        self.type = args.type

        # load concepts
        self.concept_finder = ConceptFinder(args.coco2concepts, args.level, args.unique)
        self.concepts = self.concept_finder.coco2synset

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
    def update_dataset_info(annotations, level, unique):
        if "info" not in annotations:
            annotations['info'] = {}
        annotations['info']['description'] = "COCO 2017 Dataset augmented with concepts."
        annotations['info']['concept_max_level'] = level
        annotations['info']['unique'] = unique
        annotations['info']['contributor'] = "Davide Rigoni"
        annotations['info']['date_created'] = "2021/12/15"

    @staticmethod
    def update_category_concepts(annotations, concepts):
        # for each COCO category extracts its concept and all its descendants
        for category in annotations['categories']:
            descendant = concepts[category['id']]['descendants']
            category['descendants'] = descendant

    @staticmethod
    def powerset( iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    @staticmethod
    def search_image_by_id(images, image_id):
        result = [i for i in images if i['id'] == image_id]
        assert len(result) == 1
        return result[0]

    def generate_output_file_name(self):
        prefix = self.coco_dataset.split('/')[-1].split(".")[0]
        output_file = "./datasets/{}/annotations/{}_{}.json".format(self.dataset_name, prefix, self.type)
        return output_file

    def generate_examples(self, annotations):
        """
        This function populates the new dictionary key 'concepts' to images.
        :param: annotations: COCO annotations.
        """
        # metaMapping = {i['id']: i['name'] for i in annotations['categories']}
        image_to_annotations = defaultdict(list)
        for i, ann in enumerate(annotations['annotations']):
            image_id = ann['image_id']
            image_to_annotations[image_id].append(i)
        n_images = len(image_to_annotations)

        # params initialization needed just for the powerset case
        new_annotations = []
        new_concepts = {}
        new_images = []
        images_counter = 0
        annotations_counter = 0
        for n, image_id in enumerate(image_to_annotations):
            print("Done {}/{}. ".format(n , n_images), end='\r')
            curr_annotations_indexes = image_to_annotations[image_id]   # annotations indexes
            curr_annotations = [annotations['annotations'][idx] for idx in curr_annotations_indexes]    # annotations
            # check that at least one bounding box exists
            if len(curr_annotations) > 0:
                if self.type == 'subset' or self.type == 'all' or self.type == 'subset_old' or self.type == 'all_old' :
                    # list_unique_cat = list(set({ann['category_id'] for ann in curr_annotations}))   # unique list of categories
                    list_categories = [ann['category_id'] for ann in curr_annotations]   # unique list of categories
                    selected_categories, selected_concepts = ConceptFinder.sample_categories_and_concepts(list_categories, self.concepts, self.type)
                    annos_filtered = [ann for ann in curr_annotations if ann['category_id'] in selected_categories]
                    new_annotations.extend(annos_filtered)
                    new_concepts[image_id] = selected_concepts
                else:
                    print("Not implemented yet.")
                    exit(1)
                    # # execute powerset
                    # current_image = MakeConceptDatasetFilter.search_image_by_id(annotations['images'], image_id)
                    # all_combinations = list(super().powerset(list_unique_cat))[1:]  # no the empty one
                    # for ps in all_combinations:
                    #     # new image
                    #     current_tmp_image = copy.deepcopy(current_image)
                    #     current_tmp_image['id'] = images_counter
                    #     new_images.append(current_tmp_image)
                    #     # filter out the annotations not aligned with the sampled categories
                    #     annos_filtered = copy.deepcopy([ann for ann in curr_annotations if ann['category_id'] in ps])
                    #     # deep copy needed to change annotations values
                    #     for ann in annos_filtered:
                    #         ann['image_id'] = images_counter
                    #         ann['id'] = annotations_counter
                    #         annotations_counter += 1
                    #     new_annotations.extend(annos_filtered)
                    #     images_counter += 1

        annotations['annotations'] = new_annotations
        # annotations['concepts'] = new_concepts
        # update images data
        for img in annotations['images']:
            img_id = img['id']
            if img_id in new_concepts:
                img['concepts'] = new_concepts[img_id]
            else:
                img['concepts'] = []
                print('No boxes for image {}. Set concepts to [].'.format(img_id))


        # only if we are doing powerset we need to update the images_list
        if self.type=='powerset':
            annotations['images'] = new_images
    
    def create_concept_dataset(self):
        # read coco annotation file
        annotations = MakeConceptDataset.load_dataset(self.coco_dataset)

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
        MakeConceptDataset.update_dataset_info(annotations, self.level, self.unique)
        # update concepts related to each category
        MakeConceptDataset.update_category_concepts(annotations, self.concepts)

        # add new concepts key
        annotations['concepts'] = []
        # generate syntetic examples
        self.generate_examples(annotations)

        # save new dataset
        output_file = self.generate_output_file_name()
        MakeConceptDataset.save_dataset(output_file, annotations)
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
                        help='COCO dataset file.',
                        default="./datasets/coco/annotations/instances_val2017.json",
                        type=str)
    parser.add_argument('--dataset_name', dest='dataset_name',
                        help='Name of the new dataset',
                        default="concept_coco_new",
                        type=str)
    parser.add_argument('--level', dest='level',
                        help='Levels to consider in the knowledge graph. If add_key=False, then args.level just put the list of descendants but do not affect the dataset.',
                        default=2,
                        type=int)
    parser.add_argument('--coco2concepts', dest='coco2concepts',
                        help='File regarding the links from coco classes to concepts.',
                        default="./concept/coco_to_synset.json",
                        type=str)
    parser.add_argument('--unique', dest='unique',
                        help='Generating considering unique property.',
                        default='True',
                        type=lambda x: True if x.lower() == 'true' else False)
    parser.add_argument('--type', dest='type',
                        help='Generating considering all or subsets',
                        choices=['subset', 'all', 'powerset', 'subset_old', 'subset_old_v2', 'all_old'],
                        default='subset')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    maker = MakeConceptDataset(args)
    maker.create_concept_dataset()