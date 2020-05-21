from torch.nn.functional import avg_pool2d, pad
from torch import pow, sqrt, clamp, mean
from torch.nn import Module, L1Loss
import torch
from torch import nn
from torchvision import models
from CX import CX_loss


def std_weight_map(image):
    # calculate std. of target local patch
    padded_rgb_images = pad(input=image, pad=(5, 5, 5, 5), mode='replicate')
    mean_rgb_images = avg_pool2d(input=padded_rgb_images, stride=1, kernel_size=11)
    square_mean_rgb_images = pow(input=mean_rgb_images, exponent=2)
    square_rgb_images = pow(input=padded_rgb_images, exponent=2)
    mean_square_rgb_images = avg_pool2d(input=square_rgb_images, stride=1, kernel_size=11)

    var_rgb_images = mean_square_rgb_images - square_mean_rgb_images
    std_rgb_images = sqrt(clamp(var_rgb_images, min=0))

    weights = 1 + clamp(std_rgb_images, max=0.4) / 0.2
    return weights.detach()


class StdWeightedLoss(Module):
    def __init__(self):
        super(StdWeightedLoss, self).__init__()
        self.loss_l1 = L1Loss(reduction='none').cuda()

    def forward(self, predict, target):
        loss_l1 = self.loss_l1(predict, target)
        weights = std_weight_map(target)
        loss = mean(loss_l1 * weights)

        return loss


def normalize_vgg(x):
    rgb_mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
    rgb_std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
    y = (x - rgb_mean) / rgb_std
    return y


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(2, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, indices=None):
        if indices is None:
            # indices = [2, 7, 12, 21, 30]
            indices = [21]
        out = []
        # indices = sorted(indices)
        for i in range(indices[-1]):
            x = self.vgg_pretrained_features[i](x)
            if (i + 1) in indices:
                out.append(x)

        return out

        # h_relu1 = self.slice1(X)
        # h_relu2 = self.slice2(h_relu1)
        # h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        # return out


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        # self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        # self.indices = indices or [2, 7, 12, 21, 30]
        self.weights = [1.0]
        self.indices = [21]

        '''
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None
        '''

    def forward(self, x, y):
        '''
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        '''
        x = normalize_vgg(x)
        y = normalize_vgg(y)

        x_vgg = self.vgg(x, self.indices)
        y_vgg = self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class CXLoss(VGGLoss):
    # Contextual Loss from
    # https://arxiv.org/abs/1803.02077
    def __init__(self, vgg=None, weights=None, indices=None, criterions=None):
        super(CXLoss, self).__init__(vgg, weights, indices)
        self.criterions = criterions or [CX_loss] * (len(self.weights))

    def forward(self, x, y):
        x = normalize_vgg(x)
        y = normalize_vgg(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)

        '''
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterions[i](x_vgg[i], y_vgg[i].detach())

        loss = loss[0] if loss.dim() == 1 else loss
        '''

        loss = contextual_loss(x_vgg, y_vgg)

        return loss


def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.

    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).

    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
      :param y:
      :param x:
      :param h:
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
    d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)

    # Eq (2)
    d_tilde = d / (d_min + 1e-5)

    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)

    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()  # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossR(DiscLoss):
    # RSGAN from
    # https://arxiv.org/abs/1807.00734
    def name(self):
        return 'RSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake - pred_real, 1)

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - pred_fake, 1)  # BCE_stable loss
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    # RaSGAN from
    # https://arxiv.org/abs/1807.00734
    def name(self):
        return 'RaSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss
