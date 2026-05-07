import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math  # 新增：用于棋盘格块计算
from rift_core import RIFT_no_rotation_invariance


def imread_color3(path):
    """读取图像并转为3通道RGB"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"图像不存在: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 新增：棋盘格生成函数（互补对）
def generate_checkerboard_a(img, d):
    """生成棋盘格模式A（奇数行偶数列块置零 + 偶数行奇数列块置零）"""
    img_copy = img.copy()
    m, n, p = img_copy.shape
    m_blocks = math.ceil(m / d)
    n_blocks = math.ceil(n / d)

    # 奇数行块的偶数列块置零
    for i in range(1, m_blocks + 1, 2):
        for j in range(2, n_blocks + 1, 2):
            row_start = (i - 1) * d
            row_end = min(i * d, m)
            col_start = (j - 1) * d
            col_end = min(j * d, n)
            img_copy[row_start:row_end, col_start:col_end, :] = 0

    # 偶数行块的奇数列块置零
    for i in range(2, m_blocks + 1, 2):
        for j in range(1, n_blocks + 1, 2):
            row_start = (i - 1) * d
            row_end = min(i * d, m)
            col_start = (j - 1) * d
            col_end = min(j * d, n)
            img_copy[row_start:row_end, col_start:col_end, :] = 0
    return img_copy


def generate_checkerboard_b(img, d):
    """生成棋盘格模式B（与模式A互补）"""
    img_copy = img.copy()
    m, n, p = img_copy.shape
    m_blocks = math.ceil(m / d)
    n_blocks = math.ceil(n / d)

    # 奇数行块的奇数列块置零
    for i in range(1, m_blocks + 1, 2):
        for j in range(1, n_blocks + 1, 2):
            row_start = (i - 1) * d
            row_end = min(i * d, m)
            col_start = (j - 1) * d
            col_end = min(j * d, n)
            img_copy[row_start:row_end, col_start:col_end, :] = 0

    # 偶数行块的偶数列块置零
    for i in range(2, m_blocks + 1, 2):
        for j in range(2, n_blocks + 1, 2):
            row_start = (i - 1) * d
            row_end = min(i * d, m)
            col_start = (j - 1) * d
            col_end = min(j * d, n)
            img_copy[row_start:row_end, col_start:col_end, :] = 0
    return img_copy


def FSC(cor1, cor2, change_form='affine', error_t=2):
    """随机抽样一致性滤波（简化版）"""
    M = cor1.shape[0]
    if M < 4:
        raise ValueError("匹配点数量不足")

    # 简单的RANSAC实现
    best_inliers = []
    best_H = None
    max_iter = 1000
    for _ in range(max_iter):
        idx = np.random.choice(M, 3, replace=False)
        src = cor1[idx]
        dst = cor2[idx]

        # 计算仿射变换矩阵
        H, _ = cv2.findAffineTransform(src.astype(np.float32), dst.astype(np.float32))
        H = np.vstack([H, [0, 0, 1]])  # 转为3x3矩阵

        # 验证内点
        src_hom = np.hstack([cor1, np.ones((M, 1))])
        pred = (H @ src_hom.T).T
        pred = pred[:, :2] / pred[:, 2:]
        dist = np.linalg.norm(pred - cor2, axis=1)
        inliers = np.where(dist < error_t)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

            # 早期终止
            if len(inliers) > M * 0.9:
                break

    # 用所有内点重新计算
    in1 = cor1[best_inliers]
    in2 = cor2[best_inliers]
    H, _ = cv2.findAffineTransform(in1.astype(np.float32), in2.astype(np.float32))
    H = np.vstack([H, [0, 0, 1]])

    # 计算RMSE
    src_hom = np.hstack([in1, np.ones((len(in1), 1))])
    pred = (H @ src_hom.T).T
    pred = pred[:, :2] / pred[:, 2:]
    rmse = np.sqrt(np.mean(np.sum((pred - in2) ** 2, axis=1)))

    return H, rmse, in1, in2


def image_fusion(im1, im2, H, d=32):
    """图像融合并生成配准后的棋盘格图像"""
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # 计算变换后的边界
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_trans = cv2.perspectiveTransform(corners, H)
    all_corners = np.vstack([corners_trans.reshape(-1, 2),
                             np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])])

    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)
    t = [-x_min, -y_min]
    H_trans = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]) @ H

    # 融合图像
    output_shape = (y_max - y_min, x_max - x_min)
    warped = cv2.warpPerspective(im1, H_trans, (output_shape[1], output_shape[0]))  # 配准后的SAR图像
    im2_padded = np.zeros_like(warped)
    im2_padded[t[1]:t[1] + h2, t[0]:t[0] + w2] = im2  # 原光学图像（填充后）

    # 生成配准后的棋盘格
    sar_checker = generate_checkerboard_a(warped, d)  # 配准后SAR图像棋盘格
    opt_checker = generate_checkerboard_b(im2_padded, d)  # 原光学图像棋盘格
    combined_checker = cv2.add(sar_checker, opt_checker)  # 叠加棋盘格

    # 保存结果
    cv2.imwrite("warped_sar.png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    cv2.imwrite("original_optical_padded.png", cv2.cvtColor(im2_padded, cv2.COLOR_RGB2BGR))
    cv2.imwrite("sar_checkerboard.png", cv2.cvtColor(sar_checker, cv2.COLOR_RGB2BGR))
    cv2.imwrite("optical_checkerboard.png", cv2.cvtColor(opt_checker, cv2.COLOR_RGB2BGR))
    cv2.imwrite("registered_combined_checkerboard.png", cv2.cvtColor(combined_checker, cv2.COLOR_RGB2BGR))

    # 显示棋盘格结果
    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.imshow(warped)
    plt.title("配准后的SAR图像")
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(im2_padded)
    plt.title("原光学图像（填充后）")
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(sar_checker)
    plt.title("SAR图像棋盘格")
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(combined_checker)
    plt.title("配准后SAR与光学图像叠加棋盘格")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 保存融合结果
    fusion = np.where(warped == 0, im2_padded, warped)
    cv2.imwrite("fusion_result.png", cv2.cvtColor(fusion, cv2.COLOR_RGB2BGR))
    plt.figure(figsize=(10, 8))
    plt.imshow(fusion)
    plt.title("融合结果")
    plt.axis('off')
    plt.show()

    return warped, im2_padded, combined_checker  # 返回配准后图像和棋盘格结果


def demo_single_pair(p1, p2, checkerboard_block_size=32):
    """单对图像匹配演示（包含配准后棋盘格功能）"""
    # 假设p1为SAR图像路径，p2为光学图像路径
    sar_img = imread_color3(p1)
    optical_img = imread_color3(p2)

    print("RIFT特征检测与描述")
    start_time = time.time()
    des_m1, des_m2 = RIFT_no_rotation_invariance(sar_img, optical_img, s=4, o=6, patch_size=96)

    print("最近邻匹配")
    A = des_m1['des'].astype(np.float32)
    B = des_m2['des'].astype(np.float32)
    if len(A) == 0 or len(B) == 0:
        raise RuntimeError("未检测到特征点，请调整参数或图像")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(A, B)
    matches = sorted(matches, key=lambda x: x.distance)

    matched1 = des_m1['kps'][[m.queryIdx for m in matches]]
    matched2 = des_m2['kps'][[m.trainIdx for m in matches]]

    # 去重
    _, uniq_idx = np.unique(matched2, axis=0, return_index=True)
    matched1 = matched1[uniq_idx]
    matched2 = matched2[uniq_idx]

    print("外点剔除")
    H, rmse, in1, in2 = FSC(matched1, matched2, 'affine', 2)
    print(f"内点数量: {len(in1)}, RMSE: {rmse:.4f}")
    print(f"总耗时: {time.time() - start_time:.4f}秒")

    # 显示匹配结果
    plt.figure(figsize=(12, 6))
    h1, w1 = sar_img.shape[:2]
    canvas = np.zeros((max(h1, optical_img.shape[0]), w1 + optical_img.shape[1], 3), dtype=np.uint8)
    canvas[:h1, :w1] = sar_img
    canvas[:optical_img.shape[0], w1:w1 + optical_img.shape[1]] = optical_img

    plt.imshow(canvas)
    for p, q in zip(in1, in2):
        plt.plot([p[0], q[0] + w1], [p[1], q[1]], 'y-', linewidth=0.5)
        plt.plot(p[0], p[1], 'ro', markersize=2)
        plt.plot(q[0] + w1, q[1], 'go', markersize=2)
    plt.title("特征匹配结果")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 图像融合并生成配准后棋盘格
    print("生成配准后的SAR与光学图像棋盘格...")
    image_fusion(sar_img, optical_img, H, d=checkerboard_block_size)


def demo_batch(img_folder1, img_folder2, num_pairs, img_ext='.tif'):
    """批量处理图像对（当前未使用）"""
    pass

if __name__ == "__main__":
    # 单对图像演示（修改为实际的图像路径）
    demo_single_pair(
        "D:\sourcecode\ImageRegistration\RIFT-multimodal-image-matching-main\python\sar-optical\pair3.tif",  # 替换为第一张图像的路径
        "D:\sourcecode\ImageRegistration\RIFT-multimodal-image-matching-main\python\sar-optical\pair4.tif",  # 替换为第二张图像的路径
        checkerboard_block_size=32  # 棋盘格块大小，可根据需要调整
    )