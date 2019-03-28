from os.path import join
from os import listdir
from torch import save, ge, load
from torch.nn import AvgPool2d
from util import util
from util.image import save_features, save_image, warp_image
from util.patchmatch import find_mappings, MapFunctions, optimize_mappings
from util.morph import create_vid, find_middle_mapping

class DeepFeatures:
    def __init__(self, vgg, opt, save_dir):
        # backward compitability
        opt.morph.smoothness_ratio = opt.morph.smoothness_to_fidality_ratio if 'smoothness_to_fidality_ratio' in opt.morph else opt.morph.smoothness_ratio
        opt.morph.bidir_ratio = opt.morph.bidir_ratio if 'bidir_ratio' in opt.morph else "0,0,0,0,0"

        self.opt = opt
        self.device = opt.device
        self.vgg = vgg
        self.MF = MapFunctions(self.device, opt.params.map_upsample_mode)
        self.save_dir = save_dir
        self.alphas = util.str_to_list(opt.params.style_to_content_ratio, coef=1e-1)
        self.tau = opt.params.tau
        self.search_radi = util.str_to_list(opt.params.patch_search_radi, to_int=True)
        self.patch_sizes = util.str_to_list(opt.params.patch_sizes, to_int=True)
        self.n_iters_patchmatch = opt.params.n_patchmatch_iter
        self.avg_filter = opt.params.image_avg_filter_size
        self.use_propogation = opt.params.use_propogation

        # morph
        self.thetas = util.str_to_list(opt.morph.smoothness_ratio, coef=1e-4)
        self.gammas = util.str_to_list(opt.morph.bidir_ratio, coef=1)
        self.use_mid_mapping = opt.morph.use_mid_mapping
        self.morph_segments = opt.morph.morph_segments
        self.do_twosided = opt.morph.do_twosided

        try:
            self.stop_level = opt.stop_at_level
        except:
            self.stop_level = 0


        util.mkdir(join(save_dir, 'features'), False)
        util.mkdir(join(save_dir, 'maps'), False)
        util.mkdir(join(save_dir, 'imgs'), False)
        util.mkdir(join(save_dir, 'tensors'), False)
        util.mkdir(join(save_dir, 'videos'), False)

    def get_M(self, F):
        assert (F.dim() == 4)
        F_sq = F.pow(2).float()
        F_sq -= F_sq.min()
        F_sq /= F_sq.max()
        F_sq -= self.tau
        F_sq *= (-300)
        return 1/(1+F_sq.exp()).float()
        # return ge(F_sq, self.tau).float()

    def run(self, A, Bt):
        save(A, open(join(self.save_dir, 'tensors', 'A.pickle'), 'wb'))
        save(Bt, open(join(self.save_dir, 'tensors', 'Bt.pickle'), 'wb'))

        L_start = 5
        L_end = 1

        # init L=5 feature maps
        F_A = self.vgg(A)
        F_Bt = self.vgg(Bt)

        # init At and B as A=At, B=Bt
        F_At = F_A.clone()
        F_B = F_Bt.clone()

        # init maps randomly
        b, _, h, w = F_A.size()
        mappings = {'A': self.MF.identity_map(F_A.size()).float(),
                    'B': self.MF.identity_map(F_A.size()).float()}

        # load tensors from previous training
        try:
            path = self.opt.load.dir
            level = self.opt.load.level
            mappings, F_A, F_At, F_B, F_Bt, A, At, B, Bt, L_start = load_prev_train(path, level)
            print("\nLoaded features and maps from previous train. Starting at L=%d"%L_start)
        except:
            pass

        for L in range(L_start, L_end-1, -1):
            print('\n------ Level %d ------' % L)
            patch_size = self.patch_sizes[L-2]
            search_box_radius = self.search_radi[L-1]

            print("Calculate mappings.")
            mappings = find_mappings(F_A, F_At, F_B, F_Bt, patch_size, mappings, search_box_radius, L,
                                     self.save_dir, self.n_iters_patchmatch, self.use_propogation)
            mappings = optimize_mappings(mappings, gamma=self.gammas[L-1], theta=self.thetas[L-1])

            if self.use_mid_mapping:
                print('Find middle mapping')
                mappings = find_middle_mapping(mappings, self.MF)

            print("Warp images.")
            upsampled_map_A = self.MF.upsampled_map(mappings['A'], L)
            upsampled_map_B = self.MF.upsampled_map(mappings['B'], L)
            At = warp_image(Bt, mapping=upsampled_map_A)
            B = warp_image(A, mapping=upsampled_map_B)

            print("Save warpped images.")
            save_image(B, join(self.save_dir, 'imgs', '%d_B.png'% L))
            save_image(At, join(self.save_dir, 'imgs', '%d_At.png'% L))
            save(mappings['A'], open(join(self.save_dir, 'tensors', '%d_map_a_to_b.pickle'% L), 'wb'))
            save(mappings['B'], open(join(self.save_dir, 'tensors', '%d_map_b_to_a.pickle'% L), 'wb'))

            print("Create and save morph video.")
            create_vid(A, Bt, upsampled_map_B, upsampled_map_A, self.morph_segments,
                       join(self.save_dir, 'videos'), L, MF=self.MF, do_twosided=self.do_twosided)

            if L == L_end or L == self.stop_level:
                # save final results
                create_vid(A, Bt, upsampled_map_B, upsampled_map_A, self.morph_segments,
                           self.save_dir, L, MF=self.MF, do_twosided=self.do_twosided)
                avg_filter = AvgPool2d(kernel_size=self.avg_filter, stride=1, padding=self.avg_filter//2)
                save_image(avg_filter(B), join(self.save_dir, '%d_B.png' % L))
                save_image(avg_filter(At), join(self.save_dir, '%d_At.png' % L))
                return

            FLBt_warped = warp_image(F_Bt, mappings['A'])
            FLA_warped = warp_image(F_A, mappings['B'])
            print("Deconvolve feature map Bt.")
            RL_1Bt = self.vgg.deconvolve(FLBt_warped, L - 1)
            print("Deconvolve feature map A.")
            RL_1A = self.vgg.deconvolve(FLA_warped, L - 1)

            print("Warp feature maps.")
            # get features of L-1 for A, Bt
            F_A = self.vgg(A, end=L - 1)
            F_Bt = self.vgg(Bt, end=L - 1)

            # calculate features of L-1 for At, B
            WL_1A = self.alphas[L - 2] * self.get_M(F_A)
            F_At = F_A * WL_1A + RL_1Bt * (1 - WL_1A)
            WL_1Bt = self.alphas[L - 2] * self.get_M(F_Bt)
            F_B = F_Bt * WL_1Bt + RL_1A * (1 - WL_1Bt)

            mappings = self.MF.upsample_both_maps(mappings)

            print('Save features.')
            feat_dict = {'F_B': F_B, 'F_A': F_A, 'F_Bt': F_Bt, 'F_At': F_At, 'R_Bt': RL_1Bt, 'R_A': RL_1A,
                         'W_A': WL_1A, 'W_Bt':WL_1Bt, 'map_a_to_b': mappings['A'], 'map_b_to_a': mappings['B'],
                         'B':B, 'At':At, 'A':A, 'Bt':Bt}
            save_features(feat_dict, join(self.save_dir, 'features'), L)


def load_prev_train(path, L):
    file_list = [x for x in listdir(path) if x.endswith('.pickle') and x.find(str(L))!=-1]
    file_dict = {}
    for filename in file_list:
        key = filename[2:].split('.')[0]
        file_dict[key] = load(join(path, filename))

    mappings = {'A':file_dict['map_a_to_b'], 'B':file_dict['map_b_to_a']}
    F_A = file_dict['F_A']
    F_At = file_dict['F_At']
    F_B = file_dict['F_B']
    F_Bt = file_dict['F_Bt']
    A = file_dict['A']
    B = file_dict['B']
    At = file_dict['At']
    Bt = file_dict['Bt']

    return mappings, F_A, F_At, F_B, F_Bt, A, At, B, Bt, (L-1)
