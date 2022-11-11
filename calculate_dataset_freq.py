import os
import sys
import itertools
import logging
import time
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from extra import ConceptFinder
import nltk
from nltk.corpus import wordnet as wn
import json
import argparse
import collections

current_dir = os.path.dirname(os.path.abspath(__file__))


def print_freq_per_class(DATASET_FILE,  OUTPUT_FOLDER):
    # load concepts
    print("Loading coco annotations in {} .".format(DATASET_FILE))
    with open(DATASET_FILE, 'r') as f:
        data = json.load(f)
    
    categories = {l['id']: l['name'] for l in data['categories']}
    annotations = data['annotations'] 
    counter = {k: 0 for k in categories.values()}
    for ann in annotations:
        category_id =  ann['category_id']
        category_label = categories[category_id]
        counter[category_label] += 1

    print("Number of total objects: {} .".format(sum(counter.values())))
    output_file = "{}count_{}".format(OUTPUT_FOLDER, DATASET_FILE.split('/')[-1])
    with open(output_file, 'w') as f:
        json.dump(counter, f, indent=4)
    print("File saved in {} .".format(output_file))



def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Root folder.',
                        default='{}/'.format(current_dir),
                        type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='Dataset in COCO json format.',
                        default="./datasets/tuning_coco/annotations/tuning_instances_val2017.json",
                        type=str)
    parser.add_argument('--output', dest='output',
                        help='Output folder.',
                        default='./')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print_freq_per_class(args.dataset, args.output)


# (dynamicHead) eudavider@vglogin0005:~/repository/DynamicHead$ python calculate_dataset_freq.py --dataset ./datasets/tuning_coco/annotations/tuning_instances_train2017.json
# Loading coco annotations in ./datasets/tuning_coco/annotations/tuning_instances_train2017.json .
# Number of total objects: 824303 .
# File saved in ./count_tuning_instances_train2017.json .
# (dynamicHead) eudavider@vglogin0005:~/repository/DynamicHead$ python calculate_dataset_freq.py --dataset ./datasets/tuning_coco/annotations/tuning_instances_val2017.json
# Loading coco annotations in ./datasets/tuning_coco/annotations/tuning_instances_val2017.json .
# Number of total objects: 35698 .
# File saved in ./count_tuning_instances_val2017.json .
# (dynamicHead) eudavider@vglogin0005:~/repository/DynamicHead$ python calculate_dataset_freq.py --dataset ./datasets/coco/annotations/instances_val2017.json
# Loading coco annotations in ./datasets/coco/annotations/instances_val2017.json .
# Number of total objects: 36781 .
# File saved in ./count_instances_val2017.json .
# (dynamicHead) eudavider@vglogin0005:~/repository/DynamicHead$ python calculate_dataset_freq.py --dataset ./datasets/coco/annotations/instances_train2017.json
# Loading coco annotations in ./datasets/coco/annotations/instances_train2017.json .
# Number of total objects: 860001 .
# File saved in ./count_instances_train2017.json .
