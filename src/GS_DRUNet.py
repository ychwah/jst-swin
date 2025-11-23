import torch.nn as nn
import torch
from .BasicBlocks import conv, downsample_avgpool, ResBlock, upsample_upconv, downsample_strideconv, sequential, downsample_maxpool, upsample_pixelshuffle, upsample_convtranspose


class UNetRes(nn.Module):
    """
        UNetRes: A UNet with added residuals
        param inc_nc: number of input channels
        param out_nc: number of output channels
        param nc: list of number of channels throughout downsampling -> upsampling
        param nb: number of convolutional blocks at each stage
    """

    def __init__(self, in_nc=1, out_nc=1, nc=(64, 128, 256, 512), nb=2, act_mode='E', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = sequential(
            *[ResBlock(nc[3], nc[3], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x


class StudentGrad(nn.Module):
    """
    Standard DRUNet model
    """

    def __init__(self, model_name, in_nc, out_nc):
        super().__init__()
        self.model_name = model_name
        self.model = UNetRes(in_nc=in_nc, out_nc=out_nc, nc=[64, 128, 256, 512])

    def forward(self, x, sigma):
        noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(x.device)
        x = torch.cat((x, noise_level_map), 1)
        out = self.model(x)
        return out


class GradStepSTD(nn.Module):
    """
    Gradient Step Structure-Texture decomposion
    """

    def __init__(self):
        super().__init__()
        self.name = "Gradient Step STD"
        self.student_grad = UNetRes(in_nc=2, out_nc=2, nc=[64, 128, 256, 512], nb=3, act_mode='E',
                                    downsample_mode='strideconv', upsample_mode='convtranspose')

    def calculate_grad(self, x):
        """
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :return: Dg(x), DRUNet output N(x)
        """
        x = x.float().requires_grad_()
        out = self.student_grad.forward(x)
        g = 0.5 * torch.sum((x - out).reshape((x.shape[0], -1)) ** 2)
        Dg = torch.autograd.grad(g, x, torch.ones_like(g), create_graph=True, only_inputs=True)[0]
        return Dg

    def forward(self, x):
        """
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        """
        # Dg, N, g = self.calculate_grad(x)
        Dg = self.calculate_grad(x)
        x_hat = x - Dg
        return x_hat

    def regularization(self, x):
        out = self.student_grad.forward(x)
        g = 0.5 * torch.sum((x - out).reshape((x.shape[0], -1)) ** 2)
        return g


class GradStepSingleChannel(nn.Module):
    """
    Gradient Step Structure-Texture decomposition
    """

    def __init__(self):
        super().__init__()
        self.name = "Gradient Step STD"
        self.student_grad = UNetRes(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='E',
                                    downsample_mode='strideconv', upsample_mode='convtranspose')

    def calculate_grad(self, x):
        """
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :return: Dg(x), DRUNet output N(x)
        """
        x = x.float().requires_grad_()
        out = self.student_grad.forward(x)
        g = 0.5 * torch.sum((x - out).reshape((x.shape[0], -1)) ** 2)
        Dg = torch.autograd.grad(g, x, torch.ones_like(g), create_graph=True, only_inputs=True)[0]
        return Dg, out, g

    def forward(self, x):
        """
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        """
        Dg, N, g = self.calculate_grad(x)
        x_hat = x - Dg
        return x_hat

    def regularization(self, x):
        out = self.student_grad.forward(x)
        g = 0.5 * torch.sum((x - out).reshape((x.shape[0], -1)) ** 2)
        return g
