from torchvision.models import vgg19
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.image import show_tensor
from util.util import StopTrain

# adapted from https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/architecture.py
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device, opt_deconv, feature_layer=[-1, 1, 6, 11, 20, 29]):
        super(VGGFeatureExtractor, self).__init__()
        self.device = device

        # deconv options
        self.criterion = get_crit(opt_deconv.criterion).to(device)
        self.lr = opt_deconv.lr
        self.print_freq = opt_deconv.print_freq

        # init vgg and divide into sub-modules by the maxpools
        model = vgg19(pretrained=True).eval().to(device)
        for k, v in model.named_parameters():
            v.requires_grad = False
        if type(feature_layer) is not list:
            feature_layer = [feature_layer]
        self.features = []
        for l_minus, l in zip(feature_layer[:-1], feature_layer[1:]):
            self.features.append(nn.Sequential(*list(model.features.children())[l_minus+1:l+1]))
        for v in self.features:
            v.requires_grad = False
        self.eval()

        self.A = None
        self.Bt = None


    def forward(self, x, start=0, end=-1):
        '''Applys only the vgg layers between start to end.'''
        y = self.features[start](x)
        for idx in range(start+1, end if end != -1 else len(self.features)):
            y = self.features[idx](y)
        return y


    def deconvolve(self, features, level, show_features=False):
        UPSCALE = 2
        ch_in_level = [3, 64, 128, 256, 512]
        b, _, h, w = features.size()

        # create and normalize the deconvolved features
        deconv_feature = torch.randn((b, ch_in_level[level], h * UPSCALE, w * UPSCALE)).float().to(self.device)
        deconv_feature -= deconv_feature.min()
        deconv_feature /= deconv_feature.max()
        deconv_feature.requires_grad = True

        # init optim, lr_sch, critirion
        optimizer = Adam([deconv_feature], lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=50, threshold=1e-5, cooldown=50)
        features = features.detach().to(self.device)
        max_clamp = features.max().float().item()
        stop_crit = StopTrain(delta=1e-5, patience=150)
        idx = 0

        # optimize the deconvolved features
        if show_features:
            show_tensor(features, 'features %d'%(level+1))
            show_tensor(deconv_feature, 'first guess')
        while True:
            idx += 1
            optimizer.zero_grad()
            output = self(deconv_feature, level, level + 1)
            loss = self.criterion(output, features)
            loss.backward()
            optimizer.step()
            deconv_feature.data.clamp_(0., max_clamp)  # ReLU from 0 to max value of the orig features
            if stop_crit.step(loss): break
            scheduler.step(loss)
            if idx % self.print_freq == 0:
                if show_features:
                    show_tensor(deconv_feature)
                print('## Iteration: {0:5d}, loss: {1:.5f}'.format(idx, loss.item()))
        print('## Iteration: {0:5d}, loss: {1:.5f}'.format(idx, loss.item()))
        return deconv_feature.clone().detach()


def get_crit(crit):
    if crit=='l1':
        return torch.nn.L1Loss(reduction='mean')
    elif crit=='l2':
        return torch.nn.MSELoss(reduction='mean')
    elif crit=='huber':
        return torch.nn.SmoothL1Loss(reduction='mean')
    else:
        raise ValueError('"%s" is not a valid loss function for deconvolution process.' % crit)
