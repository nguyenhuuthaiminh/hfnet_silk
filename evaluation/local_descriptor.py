import numpy as np
import cv2

from .utils.metrics import compute_pr, compute_average_precision
from .utils.descriptors import matching
from .utils.keypoints import keypoints_warp_2D, keypoints_warp_3D
from .utils.misc import to_homogeneous, angle_error

from model.export_local import export_loader
import tqdm as tq

def compute_homography_error(kpts1, kpts2, matches, shape2, H_gt):
    if matches.shape[0] == 0:
        return False, None
    kpts1 = kpts1[matches[:, 0]]
    kpts2 = kpts2[matches[:, 1]]
    H, _ = cv2.findHomography(kpts2, kpts1, cv2.RANSAC, 3.0)
    if H is None:
        return None

    w, h = shape2
    corners2 = to_homogeneous(
        np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]))
    corners1_gt = np.dot(corners2, np.transpose(H_gt))
    corners1_gt = corners1_gt[:, :2] / corners1_gt[:, 2:]
    corners1 = np.dot(corners2, np.transpose(H))
    corners1 = corners1[:, :2] / corners1[:, 2:]
    mean_dist = np.mean(np.linalg.norm(corners1 - corners1_gt, axis=1))
    return mean_dist


def compute_pose_error(kpts1, kpts2_3d_2, matches, vis1, vis2, T_2to1, K1,
                       reproj_thresh):
    valid = vis1[matches[:, 0]] & vis2[matches[:, 1]]
    matches = matches[valid]
    failure = (None, None)

    if len(matches) < 4:
        return failure

    kpts1 = kpts1[matches[:, 0]].astype(np.float32).reshape((-1, 1, 2))
    kpts2_3d_2 = kpts2_3d_2[matches[:, 1]].reshape((-1, 1, 3))
    success, R_vec, t, inliers = cv2.solvePnPRansac(
        kpts2_3d_2, kpts1, K1, np.zeros(4), flags=cv2.SOLVEPNP_P3P,
        iterationsCount=1000, reprojectionError=reproj_thresh)
    if not success:
        return failure

    R, _ = cv2.Rodrigues(R_vec)
    t = t[:, 0]

    error_t = np.linalg.norm(t - T_2to1[:3, 3])
    error_R = angle_error(R, T_2to1[:3, :3])
    return error_t, error_R


def compute_matching_score(kpts1, kpts2, kpts1_w, kpts2_w, matches1, matches2,
                           vis1, vis2, dist_thresh=5):

    def compute_matching_score_single(kpts_w, kpts, matches, vis_w):
        vis_matched = vis_w[matches[:, 0]]
        match_dist = np.linalg.norm(kpts_w[matches[:, 0]]
                                    - kpts[matches[:, 1]], axis=-1)
        correct_matches = ((match_dist < dist_thresh)*vis_matched).sum()
        match_score = correct_matches / np.maximum(np.sum(vis_w), 1.0)
        return match_score

    score1 = compute_matching_score_single(kpts1_w, kpts2, matches1, vis1)
    score2 = compute_matching_score_single(kpts2_w, kpts1, matches2, vis2)
    score = (score1 + score2) / 2
    return score


def compute_tp_fp(kpts1_w, kpts2, vis1, matches, distances, dist_thresh=3):
    all_2D_dist = np.linalg.norm(kpts1_w[vis1][:, np.newaxis]
                                 - kpts2[np.newaxis], axis=-1)
    num_gt = (all_2D_dist.min(1) < dist_thresh).sum()
    valid_matches = vis1[matches[:, 0]]  # with visible keypoints
    matches = matches[valid_matches]  # filter visible matches
    distances = distances[valid_matches]
    tp = np.linalg.norm(kpts1_w[matches[:, 0]]
                        - kpts2[matches[:, 1]], axis=-1) < dist_thresh
    fp = np.logical_not(tp)
    return num_gt, tp, fp, distances


def compute_pose_recall(errors, num_queries):
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / num_queries
    errors = np.concatenate([[0], errors])
    recall = np.concatenate([[0], recall])
    return errors, recall


def evaluate(model, dataloader, config):
    iterations = 0
    num_kpts = []
    pose_errors = []
    pose_correctness = []
    matching_scores = []
    all_tp = []
    all_num_gt = 0
    all_distances = []

    for data in tq.tqdm(dataloader):
        iterations += 1

        # print(data['image'].shape)

        shape1 = data['image'].detach().cpu().numpy().shape[-2:][::-1]  
        shape2 = data['image2'].detach().cpu().numpy().shape[-2:][::-1]  

        # print(shape1,shape2)

        pred1 = export_loader(model(data['image'].unsqueeze(0)), config, 
                              data['image'].squeeze(0))
        pred2 = export_loader(model(data['image2'].unsqueeze(0)), config, 
                              data['image2'].squeeze(0))
        

        num_kpts.extend([len(pred1['keypoints']), len(pred2['keypoints'])])
        if len(pred1['keypoints']) == 0 or len(pred2['keypoints']) == 0:
            print('No keypoints detected')
            continue
        matches1, dist1 = matching(
            pred1['local_descriptors'], pred2['local_descriptors'],
            do_ratio_test=True, cross_check=False)
        matches2, dist2 = matching(
            pred2['local_descriptors'], pred1['local_descriptors'],
            do_ratio_test=True, cross_check=False)


        H = data['homography'][0]
        kpts1_w, vis1 = keypoints_warp_2D(
            pred1['keypoints'], np.linalg.inv(H), shape2)
        kpts2_w, vis2 = keypoints_warp_2D(
            pred2['keypoints'], H, shape1)

        # print(data['name'][0],pred1['keypoints'].shape)

        error_H = compute_homography_error(
            pred1['keypoints'], pred2['keypoints'], matches1, shape2,H)
        error = {'homography': error_H}
        correct = ((error_H < config['correct_H_thresh'])
                    if error_H is not None else False)
        
        pose_correctness.append(correct)
        pose_errors.append(error)

        matching_score = compute_matching_score(
            pred1['keypoints'], pred2['keypoints'], kpts1_w, kpts2_w,
            matches1, matches2, vis1, vis2, config['correct_match_thresh'])
        matching_scores.append(matching_score)

        num_gt, tp, _, distances = compute_tp_fp(
            kpts1_w, pred2['keypoints'], vis1, matches1, dist1,
            config['correct_match_thresh'])
        # print(num_gt, tp, distances)
        all_tp.append(tp)
        all_num_gt += num_gt
        all_distances.append(distances)
        

    precision, recall, distances = compute_pr(
        np.concatenate(all_tp, 0), np.concatenate(all_distances, 0),
        all_num_gt)
    mAP = compute_average_precision(precision, recall)

    pose_errors = {k: np.array([e[k] for e in pose_errors if e[k] is not None])
                    for k in pose_errors[0]}
    pose_recalls = {k: compute_pose_recall(v, iterations)
                    for k, v in pose_errors.items()}

    metrics = {
            'average_num_keypoints': np.mean(num_kpts),
            'matching_score': np.mean(matching_scores),
            'pose_correctness': np.mean(pose_correctness),
            'mAP': mAP,
        }
    return metrics, pose_errors, pose_recalls
            