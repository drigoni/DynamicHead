#!/usr/bin/env python
"""
Created on 18/11/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: 
"""

import copy
from collections import defaultdict


def evaluation_filtering_process(coco_api, predictions, coco2synset, dataset_metadata):
    """Post-processing done using concepts in the dataset file"""
    predictions = copy.deepcopy(predictions)
    # load concept from dataset
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids) # list of dict for each image
    images_concepts = {i["id"]:i['concepts'] for i in imgs}

    # make the pool of accepted ancestors
    all_accepted_concepts = {dataset_metadata.thing_dataset_id_to_contiguous_id[k]: val_dict['descendants'] + [val_dict['synset']] for k, val_dict in coco2synset.items() }
    
    # print(predictions) list of dict
    idx_done = []
    filtered_predictions = []
    for pred in predictions:
        img_id = pred['image_id']
        img_instances = pred['instances']
        assert img_id not in idx_done
        idx_done.append(img_id)

        #  print("IMAGE ID: ", img_id)
        # all_accepted_concepts = {dataset_metadata.thing_dataset_id_to_contiguous_id[k]: val_dict['ancestors'] + [val_dict['synset']] for k, val_dict in coco2synset.items() }
        # categories_names = {dataset_metadata.thing_dataset_id_to_contiguous_id[k]: val_dict['category'] for k, val_dict in coco2synset.items() }
        #  print("Tutte le categorie:", categories_names.items())
        #  print("Concetti dell'immagine in questione:", images_concepts[img_id])
        #  set_of_cat_id = {i['category_id'] for i in  img_instances}
        #  print("Categorie nell'immagine", set_of_cat_id)
        #  print("Label categorie nell'immagine", [categories_names[i] for i in set_of_cat_id])
        #  exit(0)
        

        # select the pool of accepted classes
        poll_accepted_classes = []
        for concept in images_concepts[img_id]:
            for cat_id, descendants in all_accepted_concepts.items():
                if concept in descendants:
                    poll_accepted_classes.append(cat_id)
        
        # filtering
        filtered_list = []
        for box in img_instances:   # {'image_id': 139, 'category_id': 62, 'bbox': [5.903441905975342, 167.05685424804688, 149.45591735839844, 93.53787231445312], 'score': 0.689603865146637}
            box_cat = box['category_id']
            if box_cat in poll_accepted_classes:
                filtered_list.append(box)

        filtered_predictions.append({
            'img_id': img_id,
            'instances': filtered_list
        })

    return filtered_predictions


def inference_filtering_process(pred, input_concepts, coco2synset, dataset_metadata):
    if len(input_concepts) == 0 or (len(input_concepts) == 1 and input_concepts[0] == 'entity.n.01'):
        print("Filtering function not applied for concepts: {}. ".format(input_concepts))
        return pred
    else:
        # make the pool of accepted ancestors
        all_accepted_concepts = {dataset_metadata.thing_dataset_id_to_contiguous_id[k]: val_dict['descendants'] + [val_dict['synset']] for k, val_dict in coco2synset.items() }
        
        # select the pool of accepted classes
        poll_accepted_classes = []
        for concept in input_concepts:
            for cat_id, descendants in all_accepted_concepts.items():
                if concept in descendants:
                    poll_accepted_classes.append(cat_id)
        
        # print(pred.keys())  # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'features', 'probs'])
        assert len(pred['pred_boxes']) == len(pred['pred_classes'])

        # filtering
        filtered_list = defaultdict(list)
        for i in range(len(pred['pred_classes'])):  
            current_class = pred['pred_classes'][i]
            if current_class in poll_accepted_classes:
                for key, item in pred.items():
                    filtered_list[key].append(item[i])
        return filtered_list
