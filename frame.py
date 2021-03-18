import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from helpers import add_ones, postRt, fundamentalToRt, normalize, EssentialMatrixTransform, myjet


def extractfeature(img):
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]

            if m.distance < 32:
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    ret.append((p1, p2))

    assert(len(set(idx1))) == len(idx1)
    assert(len(set(idx2))) == len(idx2)
    assert(len(ret)) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    model, inliners = ransac((ret[:, 0], ret[:, 1]), EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=RANSAC_RESIDUAL_THRES,
                            max_trials=RANSAC_MAX_TRIALS)
    print("Matches: %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliners), sum(inliners)))
    return idx1[inliners], idx2[inliners], fundamentalToRt(model.params)


class Frame(object):

    @property
    def kps(self):
        return kps
    @property
    def kd(self):
        return kd

