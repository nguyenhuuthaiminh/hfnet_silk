import numpy as np

from .utils.keypoints import keypoints_warp_2D, keypoints_warp_3D
from .utils.misc import div0
from .utils.metrics import compute_pr, compute_average_precision

from model.export_local import export_loader
import tqdm

import matplotlib.pyplot as plt

def compute_correctness(kpts1, kpts2, kpts1_w, kpts2_w, thresh, mutual=True):

    def compute_correctness_single(kpts, kpts_w):
        dist = np.linalg.norm(kpts_w[:, np.newaxis]
                              - kpts[np.newaxis], axis=-1)
        min_dist = np.min(dist, axis=1)
        correct = min_dist <= thresh
        if mutual:
            idx = np.argmin(dist, axis=1)
            idx_w = np.argmin(dist, axis=0)
            correct &= np.equal(np.arange(len(kpts_w)), idx_w[idx])
        return min_dist, correct

    dist1, correct1 = compute_correctness_single(kpts2, kpts1_w)
    dist2, correct2 = compute_correctness_single(kpts1, kpts2_w)
    return correct1, correct2, dist1, dist2

def evaluate(model, dataloader, config):
    num_kpts = []
    loc_error = []
    repeatability = []
    all_tp = []
    all_num_gt = 0
    all_scores = []

    for data in tqdm.tqdm(dataloader):
        
        shape1 = data['image'].detach().cpu().numpy().shape[-2:][::-1]  
        shape2 = data['image2'].detach().cpu().numpy().shape[-2:][::-1]  

        # print(shape1,shape2)

        pred1 = export_loader(model(data['image'].unsqueeze(0)), config, 
                              data['image'].squeeze(0))

        pred2 = export_loader(model(data['image2'].unsqueeze(0)), config, 
                              data['image2'].squeeze(0))

        num_kpts.extend([len(pred1['keypoints']), len(pred2['keypoints'])])
        if len(pred1['keypoints']) == 0 or len(pred2['keypoints']) == 0:
            repeatability.append(0)
            continue
        H = data['homography'][0]
        # print(H.shape)
        kpts1_w, vis1 = keypoints_warp_2D(
            pred1['keypoints'], np.linalg.inv(H), shape2)
        kpts2_w, vis2 = keypoints_warp_2D(
            pred2['keypoints'], H, shape1)


        correct1, correct2, dist1, dist2 = compute_correctness(
            pred1['keypoints'], pred2['keypoints'], kpts1_w, kpts2_w,
            config['correct_match_thresh'])

        error1 = dist1[vis1 & correct1]
        if len(error1) > 0:
            loc_error.append(error1.mean())
        error2 = dist2[vis2 & correct2]
        if len(error2) > 0:
            loc_error.append(error2.mean())

        repeat = div0(correct1[vis1].sum() + correct2[vis2].sum(),
                        vis1.sum() + vis2.sum())
        repeatability.append(repeat)

        all_tp.extend([correct1[vis1], correct2[vis2]])
        all_scores.extend([pred1['scores'][vis1], pred2['scores'][vis2]])
        all_num_gt += vis2.sum() + vis1.sum()
        # break

    precision, recall, scores = compute_pr(
        np.concatenate(all_tp, 0), np.concatenate(all_scores, 0), all_num_gt,
        reverse=True)  # confidence is in decreasing order
    mAP = compute_average_precision(precision, recall)

    metrics = {
        'average_num_keypoints': np.mean(num_kpts),
        'localization_error': np.mean(loc_error),
        'repeatability': np.mean(repeatability),
        'mAP': mAP,
    }
    return metrics
