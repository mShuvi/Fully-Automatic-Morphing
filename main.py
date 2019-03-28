from os import getcwd
from os.path import join
from models.vgg19_model import VGGFeatureExtractor
from models.deep_feature_fusion import DeepFeatures
from util.image import load_images_to_tensors
from random import seed as init_seed
import argparse
from util.util import get_options_dict, mkdir_and_rename
from shutil import copyfile, make_archive
from torch import manual_seed
from torch.cuda import manual_seed_all


def main(opt):
    # init seed for reproducability
    manual_seed(opt.seed)
    init_seed(opt.seed)
    if opt.device == 'cuda':
        manual_seed_all(opt.seed)
    opt.device = 'cuda' if 'cuda' in opt.device else 'cpu'

    # create results dir
    root_dir = getcwd()
    expiriment_path = join(root_dir, opt.data.result_path, opt.name)
    mkdir_and_rename(expiriment_path)
    copyfile(opt.opt_path, join(expiriment_path, 'opt.json'))
    make_archive(join(expiriment_path, 'code'), format='zip', root_dir=root_dir)

    # init models and get images
    vgg19 = VGGFeatureExtractor(device=opt.device, opt_deconv=opt.deconv).to(opt.device)
    A, Bt = load_images_to_tensors(opt.data.imgA, opt.data.imgB, resize_im=opt.data.resize_images, device=opt.device)
    vgg19.A = A
    vgg19.Bt = Bt

    model = DeepFeatures(vgg=vgg19, save_dir=expiriment_path, opt=opt)
    model.run(A, Bt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True,
                        help='Path to option JSON file. Can be a file or a folder of JSON files.')
    opt_list = get_options_dict(parser.parse_args().opt)
    for opt in opt_list:
        from datetime import datetime
        start = datetime.now()
        main(opt=opt)
        print('\n------ Proccess time %s ------' % (datetime.now()-start))
