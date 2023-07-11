import random
from typing import *
import torch
import math
from collections import defaultdict
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def random_value(min_val: int = -1, max_val: int = 1):
    return random.uniform(min_val, max_val)


    

def relative_iou(gt_labels: torch.Tensor, _gt_bbs: torch.Tensor, pred_labels: torch.Tensor, pred_bb: torch.Tensor, pred_scores: torch.Tensor):
    score_per_label = list()


    pred_dict, gt_dict = setup_dicts(pred_labels, pred_scores, pred_bb, gt_labels, _gt_bbs)


    for label in list(pred_dict.keys()):

        # bb_id = 0
        if label in list(gt_dict.keys()):
            pred_bbs = np.array(pred_dict[label])
            # print(f'pred_bbs: {pred_bbs}')
            gt_bbs = gt_dict[label]
            # print(f'gt_bbs: {gt_bbs}')
            # print(f'label: {label}')
            for gt_bb in gt_bbs:
                # print(f'gt_bb: {gt_bb.shape}')
                # print(f'pred_bbs: {pred_bbs.shape}')
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
                # print(f'gt_bb: {gt_bb}')
                # print(f'candidate_bb: {candidate_bb}')
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
    gt_labels = torch.squeeze(gt_labels).tolist()
    # print(f'gt_labels: {gt_labels}')

    if not isinstance(gt_labels,int):
        for idx, label in enumerate(gt_labels):
            gt_bb = torch.squeeze(_gt_bbs)
            gt_dict[int(label)].append(gt_bb[idx].numpy()) 
    else: 
        gt_bb = torch.squeeze(_gt_bbs)
        gt_dict[int(gt_labels)].append(torch.squeeze(_gt_bbs).numpy())

    # print(f'pred_scores: {pred_scores}')
    if not isinstance(pred_labels,int):
        for (idx1, label), score in zip(enumerate(pred_labels.tolist()), pred_scores.tolist()):
            if score > 0.65:
                pred_dict[int(label)].append((pred_bb[idx1].numpy(), score))
    else: 
        if pred_scores > 0.65:
            pred_dict[int(pred_labels)].append((torch.squeeze(pred_bb).numpy(), float(pred_scores)))
            
    return pred_dict, gt_dict

def compute_iou(gt_bb: List[Union[float, float, float, float]], 
                pred_bb: List[Union[float, float, float, float]]):
    # get coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = extract_coordinates(gt_bb)
    pred_x1, pred_y1, pred_x2, pred_y2 = extract_coordinates(pred_bb)
    # intersection box design
    bot_left_x = max(gt_x1, pred_x1)
    bot_left_y = max(gt_y1, pred_y1)
    top_right_x = min(gt_x2, pred_x2)
    top_right_y = min(gt_y2, pred_y2)
    # print(f'intersection box: [{bot_left_x},{bot_left_y}, {top_right_x}, {top_right_y}]')

    # intersection area
    intersection = max(0, top_right_x - bot_left_x + 1) * max(0, top_right_y - bot_left_y + 1)
    # print(f'max(0, top_right_y - bot_left_y + 1): {max(0, top_right_y - bot_left_y + 1)}')


    area_gt = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    area_pred = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    
    union = (area_gt + area_pred - intersection)
    if union != 0:
        score = (intersection / union)
    else: 
        score = 0.0
    return score



def extract_coordinates(bb):
    return math.floor(bb[0]), math.floor(bb[1]), math.ceil(bb[2]), math.ceil(bb[3])

def compute_mAP(metric_setting:MeanAveragePrecision,
                gt_labels: torch.Tensor,
                gt_bb: torch.Tensor, 
                pred_labels: torch.Tensor, 
                pred_bb: torch.Tensor, 
                pred_scores: torch.Tensor):
    
    preds = [setup_pred_dict_mAP(pred_labels, pred_bb, pred_scores)]
    target = [setup_target_dict_mAP(gt_labels, gt_bb)]
    # print(f'preds: {preds}')
    # print(f'target: {target}')
    metric_setting.update(preds=preds, target=target)
    score = metric_setting.compute()
    return score

def setup_pred_dict_mAP(labels:torch.Tensor, bb:torch.Tensor, scores:torch.Tensor):
    # qui vogliamo un dict nella lista per ogni label
    tmp = dict()
    # print(f'pred_label before: {bb}')
    if len(bb.shape) == 1 and labels.nelement() != 0:
        bb = torch.reshape(bb, (1,4))

    tmp['boxes'] = bb
    tmp['labels'] = labels
    tmp['scores'] = scores

    # print(f'pred_label then: {bb}')

    return tmp

def setup_target_dict_mAP(labels:torch.Tensor, bb:torch.Tensor):
    # qui vogliamo un dict nella lista per ogni label
    tmp = dict()
    
    # print(f'before_tar: {labels}')
    if len(labels[0]) > 1:
        labels = torch.squeeze(labels)
        bb = torch.squeeze(bb)
    elif len(labels[0]) == 1 and labels.nelement() != 0: 
        labels = torch.reshape(labels, (1,))
        bb = torch.reshape(bb, (1,4))

    # bb = torch.squeeze(bb)
    # else: 
    #     print('ciao')
    #     bb = torch.unsqueeze(bb, dim=0)
    # bb = torch.squeeze(bb)
    # print(f'then_tar: {labels}')

    tmp['boxes'] = bb
    tmp['labels'] = labels
    return tmp


def compute_ratio(gt_bb: torch.Tensor,
                    pred_bb: torch.Tensor):
    
    # get coordinates
    gt_x1, gt_y1, gt_x2, gt_y2 = extract_coordinates(gt_bb)
    pred_x1, pred_y1, pred_x2, pred_y2 = extract_coordinates(pred_bb)

    area_gt = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)

    area_pred = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    ratio = area_pred / area_gt
    # print(f'ratio: {ratio}')
    return ratio