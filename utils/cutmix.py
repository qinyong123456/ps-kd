import torch
import numpy as np

def rand_bbox(size, lam):
    """
    生成裁剪区域的边界框
    :param size: 图像的尺寸 (batch_size, channels, height, width)
    :param lam: 裁剪比例
    :return: 裁剪区域的边界框坐标
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    
    # 确保裁剪尺寸至少为2，避免生成无效的裁剪框
    min_cut_size = 2
    cut_w = max(min_cut_size, int(W * cut_rat))
    cut_h = max(min_cut_size, int(H * cut_rat))
    
    # 调整中心点范围，确保裁剪框不会超出边界
    cx_min = 0
    cx_max = W - 1
    cy_min = 0
    cy_max = H - 1
    
    # 随机选择裁剪区域的中心
    cx = np.random.randint(cx_min, cx_max + 1)
    cy = np.random.randint(cy_min, cy_max + 1)

    # 计算裁剪区域的边界，确保 bbx1 < bbx2 和 bby1 < bby2
    bbx1 = max(0, cx - (cut_w - 1) // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby1 = max(0, cy - (cut_h - 1) // 2)
    bby2 = min(H, cy + cut_h // 2)
    
    # 确保边界有效性
    assert bbx1 < bbx2, f"bbx1={bbx1}, bbx2={bbx2}, cut_w={cut_w}, cx={cx}, W={W}"
    assert bby1 < bby2, f"bby1={bby1}, bby2={bby2}, cut_h={cut_h}, cy={cy}, H={H}"
    
    return bbx1, bby1, bbx2, bby2

def cutmix(data, targets, alpha=1.0):
    """
    实现 CutMix 数据增强
    :param data: 输入的图像数据 (batch_size, channels, height, width)
    :param targets: 对应的标签
    :param alpha: Beta 分布的参数
    :return: 混合后的图像数据、调整后的标签、裁剪比例
    """
    # 从 Beta 分布中采样裁剪比例
    lam = np.random.beta(alpha, alpha)
    batch_size = data.size()[0]
    index = torch.randperm(batch_size)

    # 生成裁剪区域的边界框
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    
    # 执行图像混合
    data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]

    # 调整裁剪比例以考虑实际裁剪区域
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    # 调整标签
    targets_a = targets
    targets_b = targets[index]

    return data, targets_a, targets_b, lam
