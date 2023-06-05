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

# def relative_iou(gt_labels: torch.Tensor, gt_bb: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor):
#     score_per_label = list()
#     for idx, label in enumerate(pred_labels):
#         if label in gt_labels:
#             score = compute_iou(gt_bb[idx], pred_bb[idx])
#             score_per_label.append((label, score))
#     return score_per_label
def relative_faulty_iou(gt_labels: torch.Tensor, gt_bb: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, pred_scores: torch.Tensor, golden_correspondance: dict):
    for lab_idx in list(golden_correspondance.keys()):
        label, idx = lab_idx.split('_')
    

def relative_iou(gt_labels: torch.Tensor, _gt_bbs: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, pred_scores: torch.Tensor):
    score_per_label = list()
    # save true bounding boxes corresponding to each label
    # gt_dict = defaultdict(lambda:[])
    # gt_labels = torch.squeeze(gt_labels)
    # for idx, label in enumerate(gt_labels.tolist()):
    #     gt_bb = torch.squeeze(gt_bb)
    #     gt_dict[str(label)].append(gt_bb[idx].numpy())
    # print(pred_dict.keys())

    # save predicted bounding boxes corresponding to each label
    # pred_dict = defaultdict(lambda:[])
    # for (idx1, label_pred), score, (idx2, label_gt) in zip(enumerate(pred_labels.tolist()), pred_scores.tolist(), enumerate(gt_labels.tolist())):
    #     gt_bb = torch.squeeze(gt_bb)
    #     gt_dict[str(label_gt)].append(gt_bb[idx2].numpy())
    #     if score > 0.7:
    #         pred_dict[str(label_pred)].append(pred_bb[idx1].numpy())

    pred_dict, gt_dict = setup_dicts(pred_labels, pred_scores, pred_bb, gt_labels, _gt_bbs)
    
    correspondence = dict()
    # print(gt_dict.keys())
    for label in list(pred_dict.keys()):
        # print(f'pred_dict[label]: {pred_dict[label]}')
        bb_id = 0
        if label in list(gt_dict.keys()):
            pred_bbs = np.array(pred_dict[label])
            gt_bbs = gt_dict[label]
            for gt_bb in gt_bbs:
                # compute the array-wise subtraction between the current gt_bb and each pred_bb corresponding to the same label
                distances = np.abs(gt_bb - pred_bbs)
                # print(f'pred_bb: {pred_bb}')
                # print(f'gt_bb: {gt_bb}')
                # print(f'distances: {distances}')
                # break
                # break

                # sum all distances
                buffer = np.sum(distances, axis = 1)
                # print(f'buffer: {buffer}')

                # take the array correspinding to the lowest distance from the reference gt_bb 
                candidate_idx = np.argmin(buffer)
                candidate_bb = pred_bbs[candidate_idx]
                # print(f'candidate_bb: {candidate_bb}')

                # compute the score between the nearest bb and the gt_bb
                score = compute_iou(gt_bb, candidate_bb)

                # save result
                score_per_label.append((label, score))

                lab_bb_id = str(label) + str(candidate_idx)
                correspondence[lab_bb_id] = candidate_bb

                bb_id += 1
                # set to the selected position an array with values such that it will never be chosen again
                pred_bbs[candidate_idx] = np.array([10000, 10000, 20000, 20000])
                #preds_bbs = np.delete(pred_bbs, np.argmin(buffer))
                # it can happen that the predicted bbs will not be predicted again what if i use the labels?
                # i cannot keep track of the predictions but i can keep track of the gt because the gt will be always the same
                # as soon as i have the ground truth it will result that the prediction will not be moved there will be a different one
                # then if i go for indices, the "for" that will check for all predictions, risk to go out of bounds because
                # the new 
    return score_per_label, correspondence

def setup_dicts(pred_labels, pred_scores, pred_bb, gt_labels, _gt_bbs):

    pred_dict = defaultdict(lambda:[])
    gt_dict = defaultdict(lambda:[])

    gt_labels = torch.squeeze(gt_labels)

    for (idx1, label_pred), score, (idx2, label_gt) in zip(enumerate(pred_labels.tolist()), pred_scores.tolist(), enumerate(gt_labels.tolist())):
        gt_bb = torch.squeeze(_gt_bbs)
        gt_dict[str(label_gt)].append(gt_bb[idx2].numpy())
        if score > 0.7:
            pred_dict[str(label_pred)].append(pred_bb[idx1].numpy())
            
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
    tmp = dict()
    if scores is not None:
        tmp['boxes']=bb
        tmp['scores']=scores
        tmp['labels']=labels
    else: 
        tmp['boxes']=bb
        tmp['labels']=labels
    return tmp
