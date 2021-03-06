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
    cam.set_id(0)
    opt.add_parameter(cam)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    graph_frames, graph_points = {}, {}

    for f in (local_frames if fix_points else frames):
        pose = f.pose
        se3 = g2o.SEQuat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_vertex(v_se3)

        est = v_se3.estimate()
        assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        assert np.allclose(pose[0:3, 3], est.translation())
        graph_frames[f] = v_se3

    for p in points:
        if not any([f in local_frames for f in p.frames]):
            continue

        pt = g2o.VertexSBAPointXYZ()
        pt.set_id(p.id * 2 + 1)
        pt.set_estimate(p.pt[0:3])
        pt.set_marginalized(True)
        pt.set_fixed(fix_points)
        opt.add_vertex(pt)

        for f, idx in zip(p.frames, p.idxs):
            if f not in graph_frames:
                continue
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_parameter_id(0, 0)
            edge.set_vertex(0, pt)
            edge.set_measurement(f.kps[idx])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)

