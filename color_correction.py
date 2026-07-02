"""
对生成结果进行色彩修正（颜色迁移），使其颜色分布更接近 GT。

方法：
  对每个 RGB 通道独立做直方图匹配 (Histogram Matching)：
  将结果图的累积直方图 (CDF) 映射到 GT 的 CDF 上，
  从而让结果图的每个通道的色调分布与 GT 一致。
  这是一种稳健的全图级颜色迁移方法，不会产生异常的像素值。

输出保存在原数据文件夹名 + "_colorupdate" 目录下。
"""

import os
import numpy as np
from PIL import Image

# ===== Config (从 evaluate_metrics.py 同步) =====
DATA_DIR = "/root/autodl-tmp/sky/results/rank4_interval10"
GT_DIR = "/root/Promptus/data/sky"
ITER_STEP = "01490"
TOTAL_IDS = 101  # id 00000 ~ 00130


def get_result_path(id_val):
    """同 evaluate_metrics.py 中的映射逻辑。"""
    folder = ((id_val + 9) // 10) * 10
    return os.path.join(
        DATA_DIR, f"{folder:05d}", f"iter_{ITER_STEP}_id_{id_val:05d}.png"
    )


def get_gt_path(id_val):
    return os.path.join(GT_DIR, f"{id_val:05d}.png")


def get_output_dir():
    """生成输出目录: 原文件夹名 + '_colorupdate'"""
    base_name = os.path.basename(DATA_DIR.rstrip("/"))
    parent = os.path.dirname(DATA_DIR)
    return os.path.join(parent, f"{base_name}_colorupdate")


def histogram_match_channel(src_ch, tgt_ch):
    """
    对单个通道做直方图匹配。
    src_ch, tgt_ch: 一维数组，像素值范围 [0, 1]
    返回匹配后的通道数组，范围 [0, 1]。
    """
    # 将值映射到 0~65535 用于累积直方图计算（高精度）
    src_flat = (src_ch * 65535).astype(np.int32).ravel()
    tgt_flat = (tgt_ch * 65535).astype(np.int32).ravel()

    # 计算源图和目标图的累积直方图
    src_bins = np.bincount(src_flat, minlength=65536)
    tgt_bins = np.bincount(tgt_flat, minlength=65536)

    src_cdf = src_bins.cumsum()
    tgt_cdf = tgt_bins.cumsum()

    # 归一化到 [0, 1]
    src_cdf = src_cdf / src_cdf[-1]
    tgt_cdf = tgt_cdf / tgt_cdf[-1]

    # 构建查找表：对每个源像素值，找到目标 CDF 中最接近的像素值
    # 即对于每个源值 val，找 tgt_val 使得 tgt_cdf[tgt_val] >= src_cdf[val]
    lookup = np.zeros(65536, dtype=np.uint16)
    tgt_idx = 0
    for src_val in range(65536):
        while tgt_idx < 65536 and tgt_cdf[tgt_idx] < src_cdf[src_val]:
            tgt_idx += 1
        lookup[src_val] = min(tgt_idx, 65535)

    # 应用查找表
    matched = lookup[src_flat].astype(np.float64) / 65535.0
    return matched.reshape(src_ch.shape)


def histogram_match_image(src_img, tgt_img):
    """
    对 RGB 图像做逐通道直方图匹配。
    src_img, tgt_img: (H, W, 3) in [0, 1]
    返回: (H, W, 3) in [0, 1]
    """
    matched = np.zeros_like(src_img)
    for c in range(3):
        matched[:, :, c] = histogram_match_channel(src_img[:, :, c], tgt_img[:, :, c])
    return matched


def process_one(id_val, out_dir):
    """处理单张图像的颜色修正（直方图匹配）。"""
    result_path = get_result_path(id_val)
    gt_path = get_gt_path(id_val)

    if not os.path.exists(result_path):
        return id_val, False, f"结果图不存在: {result_path}"
    if not os.path.exists(gt_path):
        return id_val, False, f"GT图不存在: {gt_path}"

    # 加载图像
    result_img = Image.open(result_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")

    # 确保尺寸一致（如果不一样则resize结果图）
    if result_img.size != gt_img.size:
        result_img = result_img.resize(gt_img.size, Image.LANCZOS)

    result_np = np.array(result_img, dtype=np.float64) / 255.0
    gt_np = np.array(gt_img, dtype=np.float64) / 255.0

    # 直方图匹配
    corrected_np = histogram_match_image(result_np, gt_np)

    # 保存结果
    out_path = os.path.join(out_dir, f"{id_val:05d}.png")
    corrected_img = Image.fromarray((corrected_np * 255).astype(np.uint8))
    corrected_img.save(out_path)

    return id_val, True, None


def main():
    out_dir = get_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    print(f"颜色修正配置:")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  GT目录:   {GT_DIR}")
    print(f"  Iter step: {ITER_STEP}")
    print(f"  IDs: 00000 ~ {TOTAL_IDS - 1:05d}")
    print(f"  方法: 逐通道直方图匹配 (Histogram Matching)")
    print()

    success = 0
    fail = 0

    for id_val in range(TOTAL_IDS):
        _, ok, err_msg = process_one(id_val, out_dir)
        if ok:
            success += 1
            print(f"  [OK] id {id_val:05d}  -> {out_dir}/{id_val:05d}.png")
        else:
            fail += 1
            print(f"  [FAIL] id {id_val:05d}: {err_msg}")

    print(f"\n完成! 成功: {success}, 失败: {fail}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
