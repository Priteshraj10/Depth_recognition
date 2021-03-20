import g2o
import numpy as np
from helpers import poseRt


def optimize(frames, points, local_window, fix_points, verbose=False, rounds=50):
    if local_window is None:
        local_frames = frames
    else:
        local_frames = frames[-local_window:]

    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
    cam = set_id(0)
    opt.add_parameter(cam)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    graph_frames, graph_points = {}, {}

    for f in (local_frames if fix_points else frames):
        pose = f.pose
        se3 = g2o.SEQuat(pose[0:3, 0:3], pose[0:3, 3])