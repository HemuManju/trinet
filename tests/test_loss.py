import torch.nn as nn
import time
from pytorch_msssim import MS_SSIM
from piq import MultiScaleSSIMLoss

from src.models.utils import calc_ssim_kernel_size, MS_SSIM_L1_LOSS


def test_ssim_loss_function(config, img1, img2):

    # Present criterion
    k = calc_ssim_kernel_size(config['image_resize'], levels=5)
    criterion_1 = MS_SSIM(win_size=k, data_range=1, size_average=True, channel=1)
    criterion_3 = nn.L1Loss()
    loss_3 = criterion_3(img1, img2)
    t = time.time()
    loss_1 = 1 - criterion_1(img1, img2) + loss_3
    print(time.time() - t)

    # Second method
    criterion_2 = MultiScaleSSIMLoss(kernel_size=k)
    t = time.time()
    loss_2 = criterion_2(img1, img2) + loss_3
    print(time.time() - t)

    print(f'Criterion 1: {loss_1} | Criterion 2: {loss_2} | {loss_3}')
