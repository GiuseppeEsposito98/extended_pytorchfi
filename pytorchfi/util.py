import random
from typing import *
import torch
import math
from collections import defaultdict
import numpy as np
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from copy import deepcopy

def random_value(min_val: int = -1, max_val: int = 1):
    return random.uniform(min_val, max_val)

# def relative_faulty_iou(gt_labels: torch.Tensor, _gt_bbs: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, 
#                         pred_scores: torch.Tensor, golden_dict: Dict['str', torch.Tensor]):
    
#     faulty_dict, gt_dict = setup_dicts(pred_labels=pred_labels, pred_scores=pred_scores, pred_bb=pred_bb, gt_labels=gt_labels, _gt_bbs=_gt_bbs)

#     score_per_label = list()
#     for G_label, bbs in golden_dict.items():
#         for t_label in list(gt_dict.keys()):
#             if G_label == t_label:
#                 for F_label in list(faulty_dict.keys()):
#                     if F_label ==t_label:
#                         fault_bbs = np.array(faulty_dict[F_label])
#                         for bb in bbs:
#                             disatnces1 = np.linalg.norm(bb[0:2] - fault_bbs[:,0:2], axis=1)
#                             disatnces2 = np.linalg.norm(bb[2:4] - fault_bbs[:,2:4], axis=1)

#                             buffer = disatnces1 + disatnces2

#                             # take the lowest one
#                             candidate_idx = np.argmin(buffer)

#                             # take the array correspinding to the lowest distance from the reference gt_bb 
#                             candidate_bb = fault_bbs[candidate_idx]

#                             # compute the score between the nearest bb and the gt_bb
#                             score = compute_iou(bb, candidate_bb)

#                             # save result
#                             score_per_label.append((F_label, score))

#                             pred_bbs = np.delete(pred_bbs, np.argmin(buffer), axis = 0)

#                             # pred_bbs[candidate_idx] = np.array([np.nan, np.nan, np.nan, np.nan])
#                             if len(pred_bbs) == 0:
#                                 break
#     return score_per_label

                            






# def relative_iou(gt_labels: torch.Tensor, gt_bb: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor):
#     score_per_label = list()
#     for idx, label in enumerate(pred_labels):
#         if label in gt_labels:
#             score = compute_iou(gt_bb[idx], pred_bb[idx])
#             score_per_label.append((label, score))
#     return score_per_label
# def relative_faulty_iou(gt_labels: torch.Tensor, gt_bb: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, pred_scores: torch.Tensor, golden_correspondance: dict):
#     for lab_idx in list(golden_correspondance.keys()):
#         label, idx = lab_idx.split('_')
    

def relative_iou(gt_labels: torch.Tensor, _gt_bbs: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, pred_scores: torch.Tensor):
    score_per_label = list()


    pred_dict, gt_dict = setup_dicts(pred_labels, pred_scores, pred_bb, gt_labels, _gt_bbs)

    correspondence = dict()

    for label in list(pred_dict.keys()):

        indices_per_label = list()
        # bb_id = 0
        if label in list(gt_dict.keys()):
            pred_bbs = np.array(pred_dict[label])
            # print(f'pred_bbs: {pred_bbs}')
            gt_bbs = gt_dict[label]
            # print(f'gt_bbs: {gt_bbs}')
            # print(f'label: {label}')
            for gt_bb in gt_bbs:
                # compute the array-wise subtraction between the current gt_bb and each pred_bb corresponding to the same label
                # distances
                #distances = np.abs(gt_bb - pred_bbs)
                disatnces1 = np.linalg.norm(gt_bb[0:2] - pred_bbs[:,0:2], axis=1)
                disatnces2 = np.linalg.norm(gt_bb[2:4] - pred_bbs[:,2:4], axis=1)
                
                # sum distances
                buffer = disatnces1 + disatnces2
                # buffer = np.sum(distances, axis = 1)
                
                # take the lowest one
                candidate_idx = np.argmin(buffer)

                # take the array correspinding to the lowest distance from the reference gt_bb 
                candidate_bb = pred_bbs[candidate_idx]

                # compute the score between the nearest bb and the gt_bb
                score = compute_iou(gt_bb, candidate_bb)

                # save result
                score_per_label.append((label, score))

                # lab_bb_id = str(label) +'_'+ str(candidate_idx)

                # correspondence[lab_bb_id] = candidate_bb

                # bb_id += 1
                # delete the already extracted array
                # questo va cambiato, si può pensare ad una lista di indici bannati (da cui non si può scegliere)
                pred_bbs = np.delete(pred_bbs, np.argmin(buffer), axis = 0)

                # pred_bbs[candidate_idx] = np.array([np.nan, np.nan, np.nan, np.nan])
                if len(pred_bbs) == 0:
                    break
    return score_per_label, pred_dict

def setup_dicts(pred_labels:torch.Tensor, pred_scores:torch.Tensor, pred_bb:torch.Tensor, gt_labels:torch.Tensor , _gt_bbs:torch.Tensor):

    pred_dict = defaultdict(lambda:[])
    gt_dict = defaultdict(lambda:[])
    gt_labels = torch.squeeze(gt_labels)

    for idx, label in enumerate(gt_labels):
        gt_bb = torch.squeeze(_gt_bbs)
        gt_dict[int(label)].append(gt_bb[idx].numpy()) 

    for (idx1, label), score in zip(enumerate(pred_labels.tolist()), pred_scores.tolist()):
        if score > 0.6:
            pred_dict[int(label)].append(pred_bb[idx1].numpy())
            
    return pred_dict, gt_dict

def compute_iou(gt_bb: List[Union[float, float, float, float]], 
                pred_bb: List[Union[float, float, float, float]]):
    # get coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = extract_coordinates(gt_bb)
    pred_x1, pred_y1, pred_x2, pred_y2 = extract_coordinates(pred_bb)
    # print(f'extract_coordinates(gt_bb): {extract_coordinates(gt_bb)}')

    # intersection box design
    bot_left_x = max(gt_x1, pred_x1)
    bot_left_y = max(gt_y1, pred_y1)
    top_right_x = min(gt_x2, pred_x2)
    top_right_y = min(gt_y2, pred_y2)
    # print(f'intersection box: [{bot_left_x},{bot_left_y}, {top_right_x}, {top_right_y}]')

    # intersection area
    intersection = max(0, top_right_x - bot_left_x + 1) * max(0, top_right_y - bot_left_y + 1)
    # print(f'intersection: {intersection}')

    # independent boxes areas
    area_gt = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    area_pred = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    union = (area_gt + area_pred - intersection)
    # print(f'union: {union}')

    score = (intersection / union)
    return score


def extract_coordinates(bb):
    return math.floor(bb[0]), math.floor(bb[1]), math.ceil(bb[2]), math.ceil(bb[2])


def extract_coordinates(bb):
    return math.floor(bb[0]), math.floor(bb[1]), math.ceil(bb[2]), math.ceil(bb[2])

def compute_mAP(metric_setting:MeanAveragePrecision,
                gt_labels: torch.Tensor,
                gt_bb: torch.Tensor, 
                pred_labels: torch.Tensor, 
                pred_bb: torch.Tensor, 
                pred_scores: torch.Tensor):
    

    
    preds = [setup_dict_mAP(pred_labels, pred_bb, pred_scores)]
    target = [setup_dict_mAP(gt_labels, gt_bb)]

    metric_setting.update(preds=preds, target=target)
    score = metric_setting.compute()
    return score

def setup_dict_mAP(labels:torch.Tensor, bb:torch.Tensor, scores:torch.Tensor=None):
    # qui vogliamo un dict nella lista per ogni label
    res = []
    tmp = dict()
    labels = torch.squeeze(labels)
    if scores is not None:
        for idx, label in enumerate(labels):
            if scores[idx] > 0.8:
                tmp['boxe']=bb[idx]
                tmp['score']=scores[idx]
                tmp['label']=label
                res.append(tmp)
        print(f'len(res): {len(res)}')
    else: 
        for idx, label in enumerate(labels):
            tmp['boxe']=bb[idx]
            tmp['label']=label
            res.append(tmp)
        print(f'len(res): {len(res)}')
    return res
