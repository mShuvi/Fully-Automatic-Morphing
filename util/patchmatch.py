import numpy as np
import torch
from util.image import save_tensor
from functools import partial
from os.path import join
from torch.nn.functional import conv2d, pad, grid_sample
from torch.nn import MSELoss
from torch.optim import Adam
from util.util import StopTrain


class TPS:
    def __init__(self, device):
        self.grad_x = torch.Tensor([[[[-1, 1], [-1, 1]]]]).to(device)
        self.grad_y = torch.Tensor([[[[-1, -1], [1, 1]]]]).to(device)
        self.lap_x = -torch.Tensor([[[[-1.,  2., -1.],
                                     [-2.,  4., -2.],
                                     [-1.,  2., -1.]]]]).to(device).requires_grad_(False)
        self.lap_y = -torch.Tensor([[[[-1., -2., -1.],
                                      [ 2.,  4.,  2.],
                                      [-1., -2., -1.]]]]).to(device).requires_grad_(False)
        self.lap_xy = -torch.Tensor([[[[-1.,  0.,  1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  0., -1.]]]]).to(device).requires_grad_(False)

    def loss(self, mapping):
        mapping = mapping.unsqueeze(1)
        by_x = conv2d(input=mapping, weight=self.lap_x) ** 2
        by_y = conv2d(input=mapping, weight=self.lap_y) ** 2
        by_xy = conv2d(input=mapping, weight=self.lap_xy) ** 2
        tps_mat = by_x + by_y + 2*by_xy
        return tps_mat.sum()

def bidir_mapping(map_a, map_b, idnty_map):
    a = ((map_a / map_a.max() - 0.5) * 2).permute([1, 2, 0]).unsqueeze(0)
    b = ((map_b / map_b.max() - 0.5) * 2).permute([1, 2, 0]).unsqueeze(0)
    warped_by_a = grid_sample(idnty_map, a, padding_mode='border')
    return grid_sample(warped_by_a, b, padding_mode='border')  # expected to be identity again

def optimize_mappings(mappings, lr=1e-2, opt_criterion=1e-6, theta=1e-6, gamma=1e1, print_freq=50):
    print("Optimize mappings with theta %g and gamma %g."%(theta, gamma))
    device = mappings['A'].device
    criterion = MSELoss(reduction='mean').to(device)
    stop_crit = StopTrain(delta=opt_criterion, patience=20)
    new_maps = {}
    optimizer = {}
    idnty_map = MapFunctions(device).identity_map(mappings['A'].size(), normalize=True).requires_grad_(False)
    idnty_map_t = idnty_map.transpose(-1, -2).unsqueeze(0).requires_grad_(False)
    theta = torch.Tensor([theta]).to(device).requires_grad_(False)
    gamma = torch.Tensor([gamma]).to(device).requires_grad_(False)

    for k in mappings.keys():
        new_maps[k] = mappings[k].clone().detach().float().requires_grad_()
        mappings[k] = mappings[k].float().requires_grad_(False)
        optimizer[k] = Adam([{'params': new_maps[k]}], lr=lr)

    tps = TPS(device)
    stop_crit.reset()

    idx = 0
    total_loss = torch.zeros(1).to(device)
    while idx < 5e4:
        idx += 1
        for this, other in zip(['A', 'B'], ['B', 'A']):
            optimizer[this].zero_grad()
            tps_loss = theta * tps.loss(new_maps[this])
            bidir_loss = gamma * criterion(bidir_mapping(new_maps[this], new_maps[other].detach(), idnty_map_t), idnty_map)
            fidelity = criterion(new_maps[this], mappings[this])
            loss = bidir_loss + tps_loss + fidelity
            loss.backward()
            total_loss += loss.detach()
            optimizer[this].step()
            if idx % print_freq == 0:
                print('## Iteration: {0:5d}, Map: {1:s}, BiDir Loss: {bid:4.4f}, TPS Loss: {tps:4.4f}, MSE: {fid:4.4f}, Loss: {loss:7.4f}'.format(idx, this, bid=bidir_loss.item(), tps=tps_loss.item(), fid=fidelity.item(), loss=loss.item()))
        if stop_crit.step(total_loss / (idx*2)): break
        if idx % print_freq == 0:
            print('## Iteration: {0:5d}, Total Loss: {loss:7.4f}'.format(idx, loss=(total_loss / idx).item()))
    for k in mappings.keys():
        new_maps[k] = new_maps[k].detach()
    print('## Iteration: {0:5d}, Total Loss: {loss:7.4f}'.format(idx, loss=(total_loss / idx).item()))
    return new_maps


class MapFunctions:
    def __init__(self, device, mode='nearest'):
        self.device = device
        self.mode = mode

    def identity_map(self, size, normalize=False):
        h, w = size[-2:]
        map_vert = torch.arange(0, w).repeat(1, h, 1).to(self.device)
        map_horiz = torch.arange(0, h).unsqueeze(1).repeat(1, 1, w).to(self.device)
        mapping = torch.cat([map_horiz, map_vert], dim=0).to(self.device).float()
        if normalize:
            mapping[0, ] /= (w-1)
            mapping[1, ] /= (h-1)
        return mapping

    def upsample(self, mapping, factor=2):
        inter = partial(torch.nn.functional.interpolate, scale_factor=factor, mode=self.mode)
        if self.mode == 'nearest':
            upsampled = inter(factor * mapping.unsqueeze(0).float()).squeeze()
        else:
            upsampled = inter(factor * mapping.unsqueeze(0).float(), align_corners=False).squeeze()
        return upsampled

    def upsampled_map(self, mapping, level):
        if level == 1:
            return mapping
        else:
            factor = 2 ** (level - 1)

            # calculate delta from the identity mapping
            delta = mapping - self.identity_map(mapping.size()).float()
            up_delta = self.upsample(delta, factor=factor)
            return self.identity_map(up_delta.size()) + up_delta

    def upsample_both_maps(self, mappings):
        new_map = {}
        for k in mappings.keys():
            new_map[k] = self.upsample(mappings[k]).clone()
            h, w = new_map[k].shape[-2:]
            new_map[k][0,].clamp_(min=0, max=h)
            new_map[k][1,].clamp_(min=0, max=w)
        return new_map


def normalize(A):
    return A / ((A.pow(2).sum(dim=1, keepdim=True)).sqrt())


def get_min_idx(tensor):
    h, w = tensor.shape[-2:]
    m = tensor.view(-1).argmin()
    return [m // w, m % w]


def get_max_idx(tensor):
    h, w = tensor.shape[-2:]
    m = tensor.view(-1).argmax()
    return [m // w, m % w]


def distance_conv(p1, p2, pt1, pt2):
    if p1.shape != p2.shape or p2.shape != pt2.shape or p2.shape != p1.shape:
        return float(0)
    return (((p1 * p2)**2).sum() + ((pt1 * pt2)**2).sum()).item()


def distance(p1, p2, pt1, pt2):
    if p1.shape != p2.shape or p2.shape != pt2.shape or p2.shape != p1.shape:
        return float('inf')
    return (((p1 - p2)**2).sum() + ((pt1 - pt2)**2).sum()).item()


def propogate(target, target_t, patch, patch_t, init_pix, mapping):
    def valid_patch(coord, h_p, w_p, h_lim, w_lim):
        h_min = max(coord[0] - h_p, 0)
        h_max = min(coord[0] + h_p + 1, h_lim)
        w_min = max(coord[1] - w_p, 0)
        w_max = min(coord[1] + w_p + 1, w_lim)
        return int(h_min), int(h_max), int(w_min), int(w_max)

    i, j = init_pix
    dh_p, dw_p = patch.shape[-2]//2, patch.shape[-1]//2
    h, w = target.shape[-2:]
    val = partial(valid_patch, h_p=dh_p, w_p=dw_p, h_lim=h, w_lim=w)
    dist = partial(distance, p2=patch, pt2=patch_t)  # TODO: should be distance_conv if using conv

    shift = {0: mapping[:, i, max(j - 1, dw_p)],
             1: mapping[:, i, min(j + 1, w-2*dw_p)],
             2: mapping[:, max(i - 1, dh_p), j],
             3: mapping[:, min(i + 1, h-2*dh_p), j],
             4: mapping[:, i, j]}

    # get adj patches coord
    adj_coord = {}
    for k in shift.keys():
        adj_coord[k] = val(shift[k])

    # get adj patches
    adj_patches = {}
    adj_patches_t = {}
    for k in shift.keys():
        adj_patches[k] = target[:, :, adj_coord[k][0]:adj_coord[k][1], adj_coord[k][2]:adj_coord[k][3]]
        adj_patches_t[k] = target_t[:, :, adj_coord[k][0]:adj_coord[k][1], adj_coord[k][2]:adj_coord[k][3]]

    # get adj dist
    dist_arr = []
    for k in shift.keys():
        dist_arr.append(dist(p1=adj_patches[k], pt1=adj_patches_t[k]))
    dist_arr = torch.Tensor(dist_arr)

    # find min distance
    min_dist, min_dist_ind = dist_arr.min(0)  # TODO: should be max if using conv
    min_dist_ind = min_dist_ind.item()
    return shift[min_dist_ind][0], shift[min_dist_ind][1], min_dist


def nnf(target, target_t, patch, patch_t, init_pix, search_radi, dx):
    # Set boundaries
    h_min = int(max(init_pix[0] - search_radi, 0))
    h_max = int(min(init_pix[0] + search_radi + 1, target.size(-2)))
    w_min = int(max(init_pix[1] - search_radi, 0))
    w_max = int(min(init_pix[1] + search_radi + 1, target.size(-1)))

    # define patch from targets
    box = target[:, :, h_min:h_max, w_min:w_max]
    box_t = target_t[:, :, h_min:h_max, w_min:w_max]

    # Find the minimal distance of a patch in the feature map
    # the distance is added between images for mutual-agreement
    with torch.no_grad():
        p_size = list(patch.shape[-2:])
        pad = (p_size[0]//2, p_size[1]//2)
        unfold = partial(torch.nn.functional.unfold, kernel_size=p_size)
        dist = (unfold(box, padding=pad) - unfold(patch))
        dist_t = (unfold(box_t, padding=pad) - unfold(patch_t))

        # mutual-agreement as squered norm2
        dist = (dist.pow(2) + dist_t.pow(2)).sum([0, 1])

        # fold
        dist = dist.view(box.shape[-2:])[dx:-dx, dx:-dx].contiguous()

    [min_h, min_w] = get_min_idx(dist)
    return [min_h + h_min + dx,
            min_w + w_min + dx,
            dist[min_h, min_w]]


def find_mappings(A, At, B, Bt, patch_size, mappings, search_radi, level, save_dir, n_iters=2, use_propogation=False):
    assert (A.size() == B.size())

    dx = patch_size // 2

    # normlize and pad edges of feature maps
    pad_ = partial(pad, pad=[dx] * 4, mode='reflect')
    imgs = [pad_(normalize(A)), pad_(normalize(B))]
    imgs_t = [pad_(normalize(At)), pad_(normalize(Bt))]

    maps_pad = {}  # pad maps to keep indices with features
    dist = {}  # distance between patches for each pixel
    for k in mappings.keys():
        maps_pad[k] = pad(mappings[k] + dx, pad=[dx] * 4)  # pad const with zeros
        dist[k] = torch.randn(imgs[0].shape[-2:]).to(mappings[k].device)

    # run only on valid pixels (non-padded)
    h_range = np.arange(A.size(-2)) + dx
    w_range = np.arange(A.size(-1)) + dx
    for curr_im, k in enumerate(mappings.keys()):
        other_im = abs(curr_im - 1)
        for i in h_range:
            for j in w_range:
                patch = imgs[curr_im][:, :, i - dx:i + dx + 1, j - dx:j + dx + 1]
                patch_t = imgs_t[curr_im][:, :, i - dx:i + dx + 1, j - dx:j + dx + 1]
                for _ in range(n_iters):
                    if use_propogation:
                        maps_pad[k][0, i, j], maps_pad[k][1, i, j], dist[k][i, j] = \
                            propogate(imgs[other_im], imgs_t[other_im], patch, patch_t,
                                      init_pix=[i, j], mapping=maps_pad[k])

                    maps_pad[k][0, i, j], maps_pad[k][1, i, j], dist[k][i, j] = \
                        nnf(imgs[other_im], imgs_t[other_im], patch, patch_t,
                            init_pix=maps_pad[k][:, i, j], search_radi=search_radi, dx=dx)

    # restore padding and indices of mappings
    for k in mappings.keys():
        mappings[k] = maps_pad[k][:, dx:-dx, dx:-dx] - dx

    if save_dir is not None:
        try:
            save_tensor(mappings['A'][0,], join(save_dir, 'maps', '%d_map_A_to_B_x' % level))
            save_tensor(mappings['A'][1,], join(save_dir, 'maps', '%d_map_A_to_B_y' % level))
            save_tensor(mappings['B'][0,], join(save_dir, 'maps', '%d_map_B_to_A_x' % level))
            save_tensor(mappings['B'][1,], join(save_dir, 'maps', '%d_map_B_to_A_y' % level))
        except:
            pass

    return mappings
