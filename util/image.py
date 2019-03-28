from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join, isfile
import torch.nn.functional as F
import numpy as np
from torch import save


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_filename(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS) and isfile(filename)


def find_min_dims(a, b):
    s1 = a.size
    s2 = b.size
    return min(s1[0], s2[0]), min(s1[1], s2[1])


def load_images_to_tensors(f_name1, f_name2, resize_im=-1, device='cpu'):
    if not is_image_filename(f_name1):
        raise TypeError('%s is not an image file.' % f_name1)
    if not is_image_filename(f_name2):
        raise TypeError('%s is not an image file.' % f_name2)
    im1 = Image.open(f_name1).convert('RGB')
    im2 = Image.open(f_name2).convert('RGB')
    h, w = find_min_dims(im1, im2)
    transforms = []
    if resize_im != -1:
        transforms.append(Resize((resize_im,resize_im),Image.BICUBIC))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms = Compose(transforms)

    return transforms(im1).to(device).unsqueeze(0), transforms(im2).to(device).unsqueeze(0)


def show_tensor(tensor, title=None):
    if len(tensor.size()) > 3:
        tensor = tensor.squeeze()
    if len(tensor.size()) == 3:
        if tensor.shape[0] == 3:
            tensor = tensor.permute([1,2,0])
        elif tensor.shape[0] > 3:
            tensor = tensor.mean(dim=0).squeeze()
    plt.imshow(tensor.detach().squeeze().cpu().numpy(), cmap='gray')
    if title is not None:
        plt.suptitle(title)
    plt.axis('off')
    plt.show()
    plt.close()


def save_tensor(tensor, path):
    tensor = tensor.squeeze()
    if tensor.shape[0] == 3:
        tensor = tensor.permute([1,2,0])
    if type(tensor) is not np.ndarray:
        tensor = tensor.detach().cpu().numpy()
    plt.imsave(path, tensor, cmap='gray')


def show_image(im):
    if len(im.shape) > 2:
        if im.shape[0] == 3:
            im = im.permute([1,2,0])
        im = im.squeeze()
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()
    # Image.fromarray(im).show()


def save_image(tensor, filename):
    image = tensor.clone()
    image -= image.min()
    image /= image.max()
    image *= 255.0
    image = image.squeeze().permute([1,2,0]).cpu().numpy()
    Image.fromarray(image.astype('uint8')).save(filename)


def save_features(kwargs, path, level):
    for k in kwargs:
        save(kwargs[k], open(join(path, '%d_%s.pickle') % (level,k), 'wb'))
        save_tensor(kwargs[k].mean(dim=1, keepdim=False).squeeze(), join(path, '%d_%s.png')%(level,k))


def warp_image(tensor, mapping):
    norm_map = mapping.float().clone()
    _, h, w = norm_map.size()
    norm_map[0,] /= (h-1)
    norm_map[1,] /= (w-1)
    norm_map -= 0.5
    norm_map *= 2

    # warp
    return F.grid_sample(tensor.transpose(-1,-2), norm_map.permute([1,2,0]).unsqueeze(0), padding_mode='border')


    # def create_norm(im_size, patch_size):
    #     h, w = im_size[-2:]
    #     with torch.no_grad():
    #         norm = torch.ones(1, 1, h, w).to(self.device)
    #         filt = torch.ones(1, 1, patch_size, patch_size).to(self.device)
    #         norm = torch.nn.functional.conv2d(input=norm, weight=filt, padding=patch_size//2)
    #     return norm.squeeze()
    #
    # warping = im[:, :, mapping[0, ], mapping[1, ]]
    # if is_image:
    #     patch = self.avg_filter
    #     warping = torch.nn.AvgPool2d(kernel_size=patch, stride=1, padding=patch//2)(warping).detach()
    # return warping #/ create_norm(im.size(), patch_size)  # TODO: check this normalization


def tensor_to_normalized_numpy(im):
    norm_im = im.clone()
    norm_im -= norm_im.min()
    norm_im /= norm_im.max()
    norm_im *= 255.0
    norm_im = norm_im.squeeze().permute([1, 2, 0]).detach().cpu().numpy().astype('uint8')
    if norm_im.shape[2] == 1:
        norm_im = np.tile(norm_im, [1, 1, 3])
    return norm_im