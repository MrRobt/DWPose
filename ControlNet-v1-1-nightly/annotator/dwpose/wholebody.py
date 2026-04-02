import os
from pathlib import Path

import cv2
import numpy as np

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose


class Wholebody:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        default_det = base_dir / 'ckpts' / 'yolox_l.onnx'
        default_pose = base_dir / 'ckpts' / 'dw-ll_ucoco_384.onnx'

        det_path = Path(os.getenv('DWPOSE_DET_PATH', str(default_det)))
        pose_path = Path(os.getenv('DWPOSE_POSE_PATH', str(default_pose)))

        use_cpu = os.getenv('DWPOSE_DEVICE', 'cuda').lower() == 'cpu'
        providers = ['CPUExecutionProvider'] if use_cpu else ['CUDAExecutionProvider']

        self.session_det = ort.InferenceSession(path_or_bytes=str(det_path), providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=str(pose_path), providers=providers)

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores
