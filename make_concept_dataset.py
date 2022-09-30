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
from extra import ConceptFinder

current_dir = os.path.dirname(os.path.abspath(__file__))


# NOTE: this class is not anymore used. It add a new key in the dataset dictionary.
class MakeConceptDataset:
    def __init__(self, args):
        # download package
        nltk.download('wordnet')
        # params
        self.coco_dataset = args.coco_dataset
        self.level = args.level
        self.unique = args.unique
        self.subset = args.subset

        # load concepts
        concept_finder = ConceptFinder(args.coco2concepts)
        self.concepts = concept_finder.extend_descendants(args.level, args.unique)

    def save_dataset(annotations_file, annotations):
        # save concept_coco annotation file
        output_file_name = annotations_file.split('/')[-1]
        output_file = "datasets/concept_coco/annotations/concept_{}".format(output_file_name)
        with open(output_file, 'w') as file:
            json.dump(annotations, file)
        print("Dataset saved in {} .".format(output_file))

    def load_dataset(annotations_file):
        print("Loading coco annotations in {} .".format(annotations_file))
        with open(annotations_file, 'r') as file:
            annotations = json.load(file)
        return annotations

    def update_dataset_info(annotations, level, unique):
        annotations['info']['description'] = "COCO 2017 Dataset augmented with concepts."
        annotations['info']['concept_max_level'] = level
        annotations['info']['unique'] = unique
        annotations['info']['contributor'] = "Davide Rigoni"
        annotations['info']['date_created'] = "2021/12/15"

    def update_category_concepts(annotations, concepts):
        # associate the right concept to each category
        # NOTE: here we can add a new class if it is needed
        # NOTE2: we use the Ivis-api coco2synsets links (clean) because many categories do not have a clear synset.
        # do it from scratch
        # for category in annotations['categories']:
        #     synset = retrieve_concept(category['name'])
        #     if category['name'] == 'mouse':
        #         synset = wn.synset('mouse.n.04')
        #     elif category['name'] == 'potted plant':
        #         synset = wn.synset('pot.n.04')
        #     print(category['supercategory'], category['name'], synset)
        # loading Ivis-api coco2synsets links

        # for each COCO category extracts its concept and all its descendants
        for category in annotations['categories']:
            # category['id'] id of the categroy
            # category['name'] name of the category
            descendant = concepts[category['id']]['descendants']
            category['descendants'] = descendant
            # print(category['descendants'])

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def _search_image_by_id(images, image_id):
        result = [i for i in images if i['id'] == image_id]
        assert len(result) == 1
        return result[0]


    def generate_examples(annotations, concepts, subset):
        image_to_annotations = defaultdict(list)
        for i, ann in enumerate(annotations['annotations']):
            image_id = ann['image_id']
            image_to_annotations[image_id].append(i)

        n_images = len(image_to_annotations)
        new_examples = []
        for n, image_id in enumerate(image_to_annotations):
            print("Done {}/{}. ".format(n , n_images), end='\r')
            curr_annotations_indexes = image_to_annotations[image_id]   # annotations indexes
            # check that at least one bounding box exists
            if len(curr_annotations_indexes) > 0:
                curr_annotations = [annotations['annotations'][idx] for idx in curr_annotations_indexes]    # annotations
                list_unique_cat = list(set({ann['category_id'] for ann in curr_annotations}))   # unique list of categories
                if subset:
                    # samples a set of categories. At least one. This or powerset
                    selected_cat = random.sample(list_unique_cat, random.randint(1, len(list_unique_cat)))
                    # filter out the annotations not aligned with the sampled categories
                    annos_filtered = [ann for ann in curr_annotations if ann['category_id'] in selected_cat]
                    # generate the associated concepts
                    gen_concepts = []
                    for ann in annos_filtered:
                        cat_idx = ann['category_id']
                        descendants = concepts[cat_idx]['descendants']
                        if len(descendants) > 0:
                            tmp_concept = random.choice(descendants)
                        else:
                            tmp_concept = concepts[cat_idx]['synset']
                        gen_concepts.append(tmp_concept)
                    tmp_dict = {
                        'image_id': image_id,
                        'concepts': gen_concepts,
                        'annotation_id': [ann['id'] for ann in annos_filtered]
                    }
                    new_examples.append(tmp_dict)
                else:
                    # execute powerset
                    all_combinations = list(powerset(list_unique_cat))[1:]  # no the empty one
                    for ps in all_combinations:
                        # filter out the annotations not aligned with the sampled categories
                        annos_filtered = [ann for ann in curr_annotations if ann['category_id'] in ps]
                        # generate the associated concepts
                        gen_concepts = []
                        for ann in annos_filtered:
                            cat_idx = ann['category_id']
                            descendants = concepts[cat_idx]['descendants']
                            if len(descendants) > 0:
                                tmp_concept = random.choice(descendants)
                            else:
                                tmp_concept = concepts[cat_idx]['synset']
                            gen_concepts.append(tmp_concept)
                        tmp_dict = {
                            'image_id': image_id,
                            'concepts': gen_concepts,
                            'annotation_id': [ann['id'] for ann in annos_filtered]
                        }
                        new_examples.append(tmp_dict)
        annotations['concept_data'] = new_examples

    def create_concept_dataset(annotations_file, concepts, level, unique, subset):
        # read coco annotation file
        annotations = load_dataset(annotations_file)

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
        update_dataset_info(annotations, level, unique)
        # update concepts related to each category
        update_category_concepts(annotations, concepts)
        # generate syntetic examples
        generate_examples(annotations, concepts, subset)

        # generate external file
        results = {}
        results['info'] = annotations['info']
        results['licenses'] = annotations['licenses']
        results['categories'] = annotations['categories']
        results['concept_data'] = annotations['concept_data']

        # save new dataset
        save_dataset(annotations_file, results)
        print("Dataset {} processed. ".format(annotations_file))
    

# NOTE: this class is the one used in the end. It filter the dataset according to the concepts it generates. It does not add any keys to the dataset dictionary.
class MakeConceptDatasetFilter(MakeConceptDataset):
    def __init__(self, args):
        super().__init__(args)

    def generate_examples(annotations, concepts, subset):
        image_to_annotations = defaultdict(list)
        for i, ann in enumerate(annotations['annotations']):
            image_id = ann['image_id']
            image_to_annotations[image_id].append(i)

        n_images = len(image_to_annotations)
        new_annotations = []
        # params initialization needed just for the powerset case
        new_images = []
        images_counter = 0
        annotations_counter = 0
        for n, image_id in enumerate(image_to_annotations):
            print("Done {}/{}. ".format(n , n_images), end='\r')
            curr_annotations_indexes = image_to_annotations[image_id]   # annotations indexes
            # check that at least one bounding box exists
            if len(curr_annotations_indexes) > 0:
                curr_annotations = [annotations['annotations'][idx] for idx in curr_annotations_indexes]    # annotations
                list_unique_cat = list(set({ann['category_id'] for ann in curr_annotations}))   # unique list of categories
                if subset:
                    # samples a set of categories. At least one. This or powerset
                    selected_cat = random.sample(list_unique_cat, random.randint(1, len(list_unique_cat)))
                    # filter out the annotations not aligned with the sampled categories
                    annos_filtered = [ann for ann in curr_annotations if ann['category_id'] in selected_cat]
                    new_annotations.extend(annos_filtered)
                else:
                    # execute powerset
                    current_image = _search_image_by_id(annotations['images'], image_id)
                    all_combinations = list(powerset(list_unique_cat))[1:]  # no the empty one
                    for ps in all_combinations:
                        # new image
                        current_tmp_image = copy.deepcopy(current_image)
                        current_tmp_image['id'] = images_counter
                        new_images.append(current_tmp_image)

                        # filter out the annotations not aligned with the sampled categories
                        annos_filtered = copy.deepcopy([ann for ann in curr_annotations if ann['category_id'] in ps])
                        # deep copy needed to change annotations values
                        for ann in annos_filtered:
                            ann['image_id'] = images_counter
                            ann['id'] = annotations_counter
                            annotations_counter += 1
                        new_annotations.extend(annos_filtered)
                        images_counter += 1

        annotations['annotations'] = new_annotations
        # only if we are doing powerset we need to update the images_list
        if not subset:
            annotations['images'] = new_images
    
    def create_concept_dataset(annotations_file, concepts, level, unique, subset):
        # read coco annotation file
        annotations = load_dataset(annotations_file)

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
        update_dataset_info(annotations, level, unique)
        # update concepts related to each category
        update_category_concepts(annotations, concepts)
        # generate syntetic examples
        generate_examples(annotations, concepts, subset)

        # save new dataset
        save_dataset(annotations_file, annotations)
        print("Dataset {} processed. ".format(annotations_file))


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Root folder.',
                        default='{}/'.format(current_dir),
                        type=str)
    parser.add_argument('--add_key', 
                        help='Generate a new dictionary of data with a ney key: "concept_data".',
                        action='store_true')
    parser.add_argument('--coco_dataset', dest='coco_dataset',
                        help='COCO dataset file.',
                        default="./datasets/coco/annotations/instances_val2017.json",
                        type=str)
    parser.add_argument('--level', dest='level',
                        help='Levels to consider in the knowledge graph.',
                        default=10,
                        type=int)
    parser.add_argument('--coco2concepts', dest='coco2concepts',
                        help='File regarding the links from coco classes to concepts.',
                        default="./concept/coco_to_synset.json",
                        type=str)
    parser.add_argument('--unique', dest='unique',
                        help='Generating considering unique property.',
                        default='True',
                        type=lambda x: True if x.lower() == 'true' else False)
    parser.add_argument('--subset', dest='subset',
                        help='Generating considering subsets or use powerset',
                        default='True',
                        type=lambda x: True if x.lower() == 'true' else False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.add_key:
        maker = MakeConceptDataset(args)
    else:
        maker = MakeConceptDatasetFilter(args)
    maker.create_concept_dataset()