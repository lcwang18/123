import numpy as np
import cv2
from phasecong3 import phasecong3


def RIFT_descriptor_no_rotation_invariance(im, kps, EO, patch_size, s, o):
    """RIFT描述符计算（无旋转不变性）"""
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=2)
    h, w = im.shape[:2]

    # 计算通道强度
    CS = np.zeros((h, w, o), dtype=np.float64)
    for j in range(o):
        acc = np.zeros((h, w), dtype=np.float64)
        for i in range(s):
            acc += np.abs(EO[i][j])
        CS[..., j] = acc
    MIM = np.argmax(CS, axis=2) + 1  # 最大强度通道图

    KPS = np.array(kps, dtype=np.float64).T  # 2xN
    ns = 6  # 6x6网格
    des = np.zeros((36 * o, KPS.shape[1]), dtype=np.float64)
    kps_to_ignore = np.zeros(KPS.shape[1], dtype=bool)

    for k in range(KPS.shape[1]):
        x = int(round(KPS[0, k]))
        y = int(round(KPS[1, k]))
        x1 = max(0, x - patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        x2 = min(x + patch_size // 2, w - 1)
        y2 = min(y + patch_size // 2, h - 1)

        if (y2 - y1) != patch_size or (x2 - x1) != patch_size:
            kps_to_ignore[k] = True
            continue

        patch = MIM[y1:y2 + 1, x1:x2 + 1]
        ys, xs = patch.shape
        hist_cube = np.zeros((ns, ns, o), dtype=np.float64)

        for jj in range(ns):
            for ii in range(ns):
                r0 = max(0, int(round(jj * ys / ns)))
                r1 = int(round((jj + 1) * ys / ns))
                c0 = max(0, int(round(ii * xs / ns)))
                c1 = int(round((ii + 1) * xs / ns))
                clip = patch[r0:r1, c0:c1].ravel()
                hist = np.bincount(np.clip(clip - 1, 0, o - 1), minlength=o)
                hist_cube[jj, ii] = hist / (np.sum(hist) + 1e-12)  # 归一化

        v = hist_cube.flatten()
        nrm = np.linalg.norm(v)
        des[:, k] = v / nrm if nrm > 0 else v

    valid = ~kps_to_ignore
    return {'kps': KPS[:, valid].T, 'des': des[:, valid].T}


def RIFT_no_rotation_invariance(im1, im2, s=4, o=6, patch_size=96):
    """RIFT特征检测与描述（无旋转不变性）"""
    # 计算相位一致性
    M1, _, _, _, _, EO1, _, _ = phasecong3(im1, nscale=s, norient=o,
                                           minWaveLength=3, mult=1.6,
                                           sigmaOnf=0.75, g=3, k=1)
    M2, _, _, _, _, EO2, _, _ = phasecong3(im2, nscale=s, norient=o,
                                           minWaveLength=3, mult=1.6,
                                           sigmaOnf=0.75, g=3, k=1)

    # 归一化到[0,1]
    def norm01(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-12)

    m1 = norm01(M1)
    m2 = norm01(M2)

    # FAST特征检测
    def detect_fast(m, max_kps=5000, min_th=10):
        img8 = np.uint8(np.clip(m * 255, 0, 255))
        fast = cv2.FastFeatureDetector_create(threshold=min_th, nonmaxSuppression=True)
        kps = fast.detect(img8, None)
        kps = sorted(kps, key=lambda kp: -kp.response)[:max_kps]
        return np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float64)

    m1_points = detect_fast(m1)
    m2_points = detect_fast(m2)

    # 计算描述符
    des_m1 = RIFT_descriptor_no_rotation_invariance(im1, m1_points, EO1, patch_size, s, o)
    des_m2 = RIFT_descriptor_no_rotation_invariance(im2, m2_points, EO2, patch_size, s, o)

    return des_m1, des_m2