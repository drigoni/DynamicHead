#!/usr/bin/env python
"""
Created on 16/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file includes the code needed for creating the new concept dataset.
"""
from collections import defaultdict
from extra import extract_COCO_concepts
import os
import argparse
import numpy as np
import json
# from extra.concept_finder_sin import ConceptFinder
from extra.concept_finder import ConceptFinder
import nltk
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
nltk.download('wordnet')
nltk.download('omw-1.4')


def check_external_concepts(concepts_folder, coco2concepts=None, concepts_depth=3):
    concepts_files = [os.path.join(concepts_folder, f) for f in os.listdir(concepts_folder) if os.path.isfile(os.path.join(concepts_folder, f))]
    res = {}
    res_count = {}
    for f in concepts_files:
        img_concepts, img_sentences = extract_COCO_concepts(f)
        res[f] = img_concepts
        res_count[f] = len(img_concepts)

    if coco2concepts is None:
        return res, res_count
    else:
        # filtering
        concept_finder = ConceptFinder(coco2concepts)
        concepts = concept_finder.extend_descendants(concepts_depth, True)
        concepts = concept_finder.extend_ancestors(concepts_depth, True)
        # # Version F
        # all_accepted_concepts = [val_dict['synset'] for k, val_dict in concepts.items()]
        # all_accepted_concepts.extend([v for k, val_dict in concepts.items() for v in val_dict['descendants']])
        # all_accepted_concepts = list(set(all_accepted_concepts))
        # res_filtered = {}
        # res_filtered_count = {}
        # for f_name, file_concepts in res.items():
        #     img_concepts_filtered = [c for c in file_concepts if c in all_accepted_concepts]
        #     res_filtered[f_name] = img_concepts_filtered
        #     res_filtered_count[f_name] = len(img_concepts_filtered)
        # return res_filtered, res_filtered_count

        # NOTE: Version Ancestors
        # there could be relations parent-child to the structures of synsets belonging to each class
        ancestors = [v for k, val_dict in concepts.items() for v in val_dict['ancestors']]
        descendants = [v for k, val_dict in concepts.items() for v in val_dict['descendants']]
        all_ancestors = list(set([v for l in ancestors for v in l]))
        all_descendants = list(set([v for l in descendants for v in l]))
        all_real_ancestors = [v for v in all_ancestors if v not in all_descendants]
        res_filtered = {}
        res_filtered_count = {}
        for f_name, file_concepts in res.items():
            img_concepts_filtered = [c for c in file_concepts if c in all_real_ancestors]
            res_filtered[f_name] = img_concepts_filtered
            res_filtered_count[f_name] = len(img_concepts_filtered)
        return res_filtered, res_filtered_count

        # # NOTE: Version P
        # # there could be relations parent-child to the structures of synsets belonging to each class.
        # # so we consider all its parents.
        res_filtered = {}
        res_filtered_count = {}
        # # code for finding ancestors for each class
        # ancestors = [v for k, val_dict in concepts.items() for v in val_dict['ancestors']]
        # descendants = [v for k, val_dict in concepts.items() for v in val_dict['descendants']]
        # relations = {k: [] for k in concepts.keys()}
        # for cls_name in concepts.keys():
        #     synset = concepts[cls_name]['synset']
        #     anc = concepts[cls_name]['ancestors']
        #     desc = concepts[cls_name]['descendants']
        #     for d in desc:
        #         if d in relations.keys():
        #            relations[synset].append(d)
        # # code for including all ancestors entities
        # for f_name, file_concepts in res.items():
        #     all_conc = []
        #     for conc in file_concepts:
        #         parents = []
        #         for cat_idx in concepts.keys():
        #             cls_descendants = concepts[cat_idx]['descendants']
        #             cls_synset = concepts[cat_idx]['synset']
        #             if conc in cls_descendants:
        #                 all_conc.append(cls_synset)
        #     all_conc = list(set(all_conc))
        #     res_filtered[f_name] = all_conc
        #     res_filtered_count[f_name] = len(all_conc)
        # return res_filtered, res_filtered_count


def check_dataset_concepts(dataset_file):
    with open(dataset_file, 'r') as f:
        annotations = json.load(f)
    res = defaultdict(list)
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        res[image_id].append(ann['category_id'])
    # filtering according to the model applied function in extra/concept_mapper.py
    res = {k: list(set(v)) for k, v in res.items() }
    res_count = {k: len(v) for k, v in res.items() }
    return res, res_count


def calculate_coverage(concepts_folder, dataset_file, coco2concepts=None, concepts_depth=3, ewiser_topk=1, check_sentences=False):
    # load data GT from dataset
    with open(dataset_file, 'r') as f:
        annotations = json.load(f)
    GTs = defaultdict(list)
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        image_name = [img['file_name'] for img in annotations['images'] if img['id'] == image_id]
        assert len(image_name) == 1
        image_name = image_name[0]
        GTs[image_name].append(ann)
    # load concepts
    concepts = {}
    sentences = {}
    for img in GTs.keys():
        concept_path = os.path.join(concepts_folder, "{}.json".format(img[:-4]))
        conc, sent = extract_COCO_concepts(concept_path, ewiser_topk)
        concepts[img] = conc
        sentences[img] = sent
    # load concepts. Not sure about ids, so change the dictionary to use category as key. 
    concept_finder = ConceptFinder(coco2concepts)
    all_concepts = concept_finder.extend_descendants(concepts_depth, True)
    all_concepts = {v['category']: v for k, v in all_concepts.items()}
    all_accepted_concepts = [val_dict['synset'] for k, val_dict in all_concepts.items()]
    all_accepted_concepts.extend([v for k, val_dict in all_concepts.items() for v in val_dict['descendants']])
    all_accepted_concepts = list(set(all_accepted_concepts))

    # print GTs category and concepts.
    categories_ids_to_labels = {i['id']: i['name']for i in annotations['categories']}
    all_hits = defaultdict(list)
    gt_synset_found = []
    gt_synset_not_found = []
    for img in GTs.keys():
        img_gts = GTs[img]
        img_boxes = ["{}:{}".format(categories_ids_to_labels[ann['category_id']], ann['bbox']) for ann in img_gts]
        img_categories_id = list(set([ann['category_id'] for ann in img_gts]))
        img_categories_labels = [categories_ids_to_labels[i] for i in img_categories_id]
        img_categories_synsets = [all_concepts[i]['synset'] for i in img_categories_labels]
        img_concepts = concepts[img]
        img_concepts_filtered = [i for i in img_concepts if i in all_accepted_concepts]
        img_sentences = sentences[img]
        for gt_label in img_categories_labels:
            find = False
            if check_sentences == True:
                # check sentences
                for gt_sentence in img_sentences:
                    if gt_label in gt_sentence:
                        find = True
                        break 
            else:
                # check synset
                for gt_descendant in all_concepts[gt_label]['descendants']:
                    if gt_descendant in img_concepts:
                        find = True
                        break 
            if find:
                all_hits[img].append(1)
                gt_synset_found.append(gt_label)
            else:
                all_hits[img].append(0)
                gt_synset_not_found.append(gt_label)
        if 'chair' in img_categories_labels and img != '000000554002.jpg':
            img_sentences = sentences[img]
            print('------------------------------------------------------')
            print("http://images.cocodataset.org/val2017/{}".format(img))
            print("#Bounding boxes: ", len(img_gts))
            print("GTs boxes: ", img_boxes)
            print("GTs labels: ", img_categories_labels)
            print("GTs synsets: ", img_categories_synsets)
            print("Not HITS: ", [label for h, label in zip(all_hits[img], img_categories_labels) if h==0])
            print("Concepts: ", img_concepts)
            print("Concepts filtered: ", img_concepts_filtered)
            print("Sentences: ")
            for i in range(len(img_sentences)):
                print("({}) {}".format(i+1, img_sentences[i]))
            exit(1)


    nposs = {k: float(len(v)) for k, v in all_hits.items()}
    accuracies = {k: np.mean(v) for k, v in all_hits.items()}
    weights = np.array(list(nposs.values()))
    weights /= weights.sum()
    print('Mean Accuracy = {:.4f}'.format(np.mean(list(accuracies.values()))))
    print('Weighted Mean Accuracy = {:.4f}'.format(np.average(list(accuracies.values()), weights=weights)))
    gt_found = dict(Counter(gt_synset_found))
    gt_found = dict(sorted(gt_found.items(), key=lambda x: x[1], reverse=True))
    print('GT found ({}) = {}'.format(len(gt_found), gt_found))
    gt_not_found = dict(Counter(gt_synset_not_found))
    gt_not_found = dict(sorted(gt_not_found.items(), key=lambda x: x[1], reverse=True))
    print('GT not found ({}) = {}'.format(len(gt_not_found), gt_not_found))
    set_diff = dict(sorted({k: v for k, v in gt_not_found.items() if k not in gt_found.keys()}.items(), key=lambda x: x[1], reverse=True))
    print('GT not found / GT found ({}) = {}'.format(len(set_diff), set_diff))
    assert len(GTs) == len(concepts)


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--folder', dest='folder',
                        help='Concepts folder.',
                        default='{}/datasets/ewiser_concepts_COCO_valid/'.format(current_dir),
                        type=str)
    parser.add_argument('--file', dest='file',
                    help='Dataset.',
                    # default='{}/datasets/coco/annotations/instances_val2017.json'.format(current_dir),
                    default=None,
                    type=str)          
    parser.add_argument('--coco2concepts', dest='coco2concepts',
                        help='File regarding the links from coco classes to concepts.',
                        default=None,
                        type=str)
    parser.add_argument('--depth', dest='depth',
                    help='Depth to consider in the knowledge graph.',
                    default=3,
                    type=int)
    parser.add_argument('--topk', dest='topk',
                help='TopK entities to consider from EWISER predictions.',
                default=3,
                type=int)
    parser.add_argument('--check_sentences', dest='check_sentences',
            help='Use sentences and words instead of concepts during coverage calculation.',
            action="store_true")
    parser.set_defaults(check_sentences=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # http://images.cocodataset.org/val2017/000000289343.jpg
    args = parse_args()
    if args.folder is not None and args.file is not None:
        print("Make comparisons.")
        calculate_coverage(args.folder, args.file, 
                            coco2concepts=args.coco2concepts, 
                            concepts_depth=args.depth, 
                            ewiser_topk=args.topk, 
                            check_sentences=args.check_sentences)
    else:
        if args.folder is not None and args.file is None:
            print("Check external concepts.")
            concepts, concepts_count = check_external_concepts(args.folder, args.coco2concepts, args.depth)
        elif args.folder is None and args.file is not None:
            print("Check dataset concepts.")
            concepts, concepts_count = check_dataset_concepts(args.file)
        else:
            print("Error. Check input parameters.")
        print("Number of images: ", len(concepts))
        print("Mean number of concepts: ", np.mean(list(concepts_count.values())))
        print("Max number of concepts: ", max(list(concepts_count.values())))
        print("Min number of concepts: ", min(list(concepts_count.values())))
        print("Number of images with 0 concepts: ", list(concepts_count.values()).count(0))