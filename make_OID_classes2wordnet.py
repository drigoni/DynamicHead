import nltk
from nltk.corpus import wordnet as wn
import argparse
import re
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
# download wordnet data
nltk.download('wordnet')


def save_json(output_file, data, indent=2):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=indent)
    print("File saved in {} .".format(output_file))


def get_synset(word):
    # remove space
    if ' ' in word:
        word = word.replace(' ', '_')
    synset = wn.synsets(word)
    if len(synset) > 0:
        synset_name = str(synset[0].name())
        # padding as yago with 9 number
        if ".n." in synset_name:
            return synset_name
        else:
            return None
    else:
        return None


def make_classes2concepts(class_list):
    # load from file
    with open(args.file_classes, 'r') as f:
        classes_list = [i.strip() for i in f.readlines()]
    # class_list = ['/m/08dz3q,Auto part', '/m/08hvt4,Jug', ...]
    classes2concepts = dict()
    for label_id, text in enumerate(classes_list):
        freebase_id = text.split(',')[0]
        label = text.split(',')[1]
        synset = get_synset(label)
        tmp_dict = {
            "coco_cat_id": label_id + 1, # start from 1 and not 0
            "meaning": label,
            "synset": synset
        }
        classes2concepts[freebase_id] = tmp_dict
    # print(classes2concepts)
    save_json(args.output, classes2concepts, 2)

    # statistics
    print("Number of missing synsets: ", len([i for i in classes2concepts.values() if i['synset'] == None ]))


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Root folder.',
                        default='{}/'.format(current_dir),
                        type=str)
    parser.add_argument('--file_classes', dest='file_classes',
                        help='File containing the list of the classes.',
                        default="./datasets/OpenImagesDataset/annotations/class-descriptions-boxable.csv",
                        type=str)
    parser.add_argument('--output', dest='output',
                    help='Output file.',
                    default="./concept/oid_to_synset.json",
                    type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    make_classes2concepts(args)





# 70 nouns are not found and need to be added manually.
# Some are very strange synsets:
# 1) hand blower and hair blower are the same synset
# 2) "Slow cooker", "jacuzzi, "Indoor rower", "Dog bed", "Cat furniture", "Bathroom accessory" and "Whiteboard"