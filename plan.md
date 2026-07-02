## Plan: 改善 CLIP loss 导致的颜色偏差问题

### TL;DR
CLIP 语义损失改善了形状轮廓但引入了全局偏色（CLIP embedding 对颜色不敏感）。计划引入多种颜色感知损失来补偿，包括：颜色统计损失、直方图损失、LAB 空间损失、HSV 色相损失、权重调优、以及可学习颜色适配层。分为三个阶段实施，每阶段可独立验证。

---

## 阶段一：基础颜色损失（推荐首选，1-2 个方案）

### 方案 1A：Color Statistics Loss（颜色统计损失）
**思路**：全局偏色的本质是生成图像的 RGB 各通道均值和标准差偏离了 GT。直接约束这两个统计量，计算量极小。

**实现**：在 `inversion.py` 的 loss 计算区域，添加：
```python
def color_stats_loss(pred, gt):
    """pred, gt: [1, 3, H, W] in [-1, 1]"""
    pred_mean = pred.mean(dim=[2, 3])   # [1, 3]
    pred_std = pred.std(dim=[2, 3])     # [1, 3]
    gt_mean = gt.mean(dim=[2, 3])
    gt_std = gt.std(dim=[2, 3])
    return F.mse_loss(pred_mean, gt_mean) + F.mse_loss(pred_std, gt_std)
```
- 新增命令行参数 `-color_weight`（默认 0.0，建议起始值 0.1~0.5）
- 几乎零额外计算开销
- 直接针对「全局偏色」问题

### 方案 1B：RGB Histogram Loss（直方图损失）
**思路**：如果仅均值和标准差不足以约束颜色分布，用可微分的直方图匹配损失强制整个颜色分布对齐。

**实现**：使用核密度估计（高斯核 soft assignment）构建可微分直方图：
```python
def histogram_loss(pred, gt, num_bins=64):
    loss = 0.0
    bins = torch.linspace(-1, 1, num_bins, device=pred.device)
    bw = bins[1] - bins[0]
    for c in range(3):
        p = pred[:, c:c+1, :, :].reshape(pred.shape[0], -1)  # [1, H*W]
        g = gt[:, c:c+1, :, :].reshape(gt.shape[0], -1)      # [1, H*W]
        # Soft histogram assignment
        p_dist = torch.exp(-((p.unsqueeze(-1) - bins.unsqueeze(0).unsqueeze(0)) / bw)**2)
        g_dist = torch.exp(-((g.unsqueeze(-1) - bins.unsqueeze(0).unsqueeze(0)) / bw)**2)
        p_hist = p_dist.mean(dim=1)   # [1, num_bins]
        g_hist = g_dist.mean(dim=1)   # [1, num_bins]
        loss = loss + F.mse_loss(p_hist, g_hist)
    return loss / 3.0
```
- 新增命令行参数 `-hist_weight`（默认 0.0，建议起始值 0.05~0.2）
- 计算成本中等，但颜色约束更强

**计划**：先实现方案 1A，如果效果不足够再叠加方案 1B。

---

## 阶段二：高级颜色损失（与阶段一可并行探索）

### 方案 2A：LAB 颜色空间 MSE Loss
**思路**：RGB 空间的欧氏距离与感知差异不线性相关，LAB 空间设计为感知均匀。将图像转到 LAB 后计算 MSE。

**实现**：使用 kornia 的 `color.rgb_to_lab()` 进行可微分的 RGB→LAB 转换：
```python
import kornia
def lab_mse_loss(pred, gt):
    # pred, gt: [1, 3, H, W] in [-1, 1]
    pred_rgb = (pred + 1) / 2  # [-1,1] -> [0,1]
    gt_rgb = (gt + 1) / 2
    pred_lab = kornia.color.rgb_to_lab(pred_rgb)
    gt_lab = kornia.color.rgb_to_lab(gt_rgb)
    return F.mse_loss(pred_lab, gt_lab)
```
- 新增命令行参数 `-lab_weight`（建议起始值 0.1~0.5）
- 因为 LAB 的 L\* 通道编码亮度，a\*/b\* 编码颜色对立信息，能更精细地约束颜色

### 方案 2B：HSV Hue-Saturation Loss
**思路**：只对 HSV 空间的 H（色相）和 S（饱和度）通道计算损失，忽略 V（明度），专门纠正颜色漂移。

**实现**：使用 kornia 的 `color.rgb_to_hsv()`：
```python
def hsv_color_loss(pred, gt):
    pred_rgb = (pred + 1) / 2
    gt_rgb = (gt + 1) / 2
    pred_hsv = kornia.color.rgb_to_hsv(pred_rgb)
    gt_hsv = kornia.color.rgb_to_hsv(gt_rgb)
    # H: [0, 2π], S: [0, 1], V: [0, 1]
    # 只约束 H 和 S 通道
    h_loss = torch.mean(torch.abs(torch.sin(pred_hsv[:, 0:1] - gt_hsv[:, 0:1])))
    s_loss = F.mse_loss(pred_hsv[:, 1:2], gt_hsv[:, 1:2])
    return h_loss + s_loss
```
- 对 H 通道使用 sin 差异来处理角度周期性
- 新增命令行参数 `-hsv_weight`（建议起始值 0.1~0.3）

---

## 阶段三：损失权重调优和架构改进

### 方案 3A：损失权重自动调优
**思路**：当前 `clip_weight=0.5` 可能过强。实验不同权重组合：
- `clip_weight` 从 0.5 → 0.1~0.3（降低 CLIP 的语义拉力）
- `temp_weight` 从 0.1 → 0.05（时序方向一致性也可能干扰颜色）
- 增加 MSE 的相对权重（从 0.8 → 进一步增大）

### 方案 3B：可学习的颜色适配层（Learnable Color Adaptation）
**思路**：在 decoder 输出后加一个小型可学习模块，专门校正颜色。

**实现**：在 `samples_x = decoder(samples_z)` 之后添加：
```python
# 一个简单的逐通道仿射变换（1x1 conv 或 per-channel scale+bias）
color_adapt = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=1, bias=True)  # 3×3 颜色变换矩阵 + bias
)
# 初始化为恒等变换
nn.init.eye_(color_adapt[0].weight)
nn.init.zeros_(color_adapt[0].bias)
color_adapt = color_adapt.cuda()
# 在 loss 计算前：samples_x = color_adapt(samples_x)
```
- 新增参数极小（12个参数），几乎不影响计算
- 单独训练或与 U/V 联合训练均可
- 可选项：约束变换矩阵接近单位矩阵（添加正则项）

### 方案 3C：后处理级联（post-training）
**思路**：训练完成后，对生成结果用 `color_correction.py` 做直方图匹配。虽然用户希望训练时解决，但可以作为保底手段。

---

## 实施路线图

```
第1步 [阶段一]：实现 Color Statistics Loss + 命令行参数
    ↓
第2步 [阶段一]：实验 color_weight=[0.1, 0.3, 0.5]，与 baseline 对比 PSNR/SSIM/LPIPS
    ↓
第3步 [可选]：若效果不足，叠加 Histogram Loss
    ↓
第4步 [阶段二]：并行探索 LAB Loss / HSV Loss
第5步 [阶段三]：权重调优实验（clip_weight, mse_weight 等）
第6步 [阶段三]：如仍不满意，实现可学习颜色适配层
```

每个步骤都可以独立验证（对比 PSNR/SSIM/LPIPS 指标）。

---

## 修改文件

| 文件 | 修改内容 |
|------|----------|
| `inversion.py` | 新增 color_stats_loss / histogram_loss / lab_mse_loss / hsv_loss 函数；修改 loss 计算逻辑；新增命令行参数 |
| `generation.py` | 无需修改（推理时权重已固化在 prompt 中） |
| `evaluate_metrics.py` | 无需修改（评估流程不变，改变的是生成质量） |

---

## 验证方法

1. 训练后在 `evaluate_metrics.py` 中评估 PSNR / SSIM / LPIPS
2. 与 `_baseline` 版本对比
3. 目视检查生成图像的颜色偏差是否改善
4. 可选：用 `color_correction.py` 做后处理，看颜色修正后指标能否进一步提升

---

## 决策记录

- 问题确定为「全局偏色」而非局部颜色不准确
- 优先在训练时通过修改 loss 解决
- 愿意尝试多种方案

## 待澄清问题

- 无（用户需求已明确）
