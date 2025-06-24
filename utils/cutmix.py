import torch
import torchvision.transforms.functional as F
import random

class Cutout(object):
    def __init__(self, n_holes, length):
        """
        初始化 Cutout 类
        :param n_holes: 要在图像中创建的孔洞数量
        :param length: 每个孔洞的边长（正方形）
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        对输入图像应用 Cutout 数据增强
        :param img: 输入的图像（PIL Image 或 Tensor）
        :return: 应用 Cutout 后得到的图像
        """
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)

        for n in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img
