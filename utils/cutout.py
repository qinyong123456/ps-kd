# utils/cutout.py
import torch
import numpy as np
import random

class Cutout:
    """
    对图像应用Cutout数据增强：随机遮挡图像中的矩形区域
    """
    def __init__(self, n_holes=1, length=16, fill_value=0):
        """
        初始化Cutout参数
        :param n_holes: 遮挡区域数量
        :param length: 遮挡区域边长（正方形）
        :param fill_value: 遮挡区域填充值（通常为0或图像均值）
        """
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value

    def __call__(self, img):
        """
        对输入图像应用Cutout
        :param img: PyTorch张量图像 (C, H, W)
        :return: 应用Cutout后的图像
        """
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for _ in range(self.n_holes):
            # 随机生成遮挡区域中心
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            
            # 计算遮挡区域边界（确保不超出图像）
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            # 应用遮挡
            mask[y1:y2, x1:x2] = self.fill_value
        
        # 扩展掩码到所有通道
        mask = mask.expand_as(img)
        img = img * mask
        
        return img
