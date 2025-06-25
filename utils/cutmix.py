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
    
    # 修正变量名拼写错误
    assert 0 <= bbx1 < bbx2 <= data.size(2), f"bbx1={bbx1}, bbx2={bbx2}, W={data.size(2)}"
    assert 0 <= bby1 < bby2 <= data.size(3), f"bby1={bby1}, bby2={bby2}, H={data.size(3)}"
    
    # 执行图像混合
    data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]

    # 调整裁剪比例以考虑实际裁剪区域
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))

    # 调整标签
    targets_a = targets
    targets_b = targets[index]

    return data, targets_a, targets_b, lam
