from util.image import warp_image, save_tensor, tensor_to_normalized_numpy
from os.path import join
from imageio import get_writer
from functools import partial


def quantize_map(field, n_segments, idx):
    return field * (float(idx) / float(n_segments))


def mix_frames_and_write(direction, warped_frames, writer, n_segments, save_dir, level):
    middle = n_segments // 2

    if direction == 'a_to_b':
        alpha = 0
        increment = float(1 / n_segments)
        path = join(save_dir, "mixed_%s_%d.png" % ('AtoB', level))
    else:
        alpha = 1
        increment = -float(1 / n_segments)
        path = join(save_dir, "mixed_%s_%d.png" % ('BtoA', level))

    for n, (warped_A, warped_B) in enumerate(zip(warped_frames['A'], warped_frames['B'])):
        mixed_frame = (1 - alpha) * warped_A + alpha * warped_B
        mixed_frame = tensor_to_normalized_numpy(mixed_frame)
        writer.append_data(mixed_frame)
        alpha += increment
        if n == middle:
            save_tensor(mixed_frame, path)


def create_vid(A, B, map_a_to_b, map_b_to_a, n_segments, save_dir, level, MF, fps=24, do_twosided=False):
    writer = get_writer(join(save_dir, "vid_" + str(level) + ".mp4"), fps=fps)

    idty_map = MF.identity_map(map_a_to_b.size())
    mappings = {'A': map_a_to_b, 'B': map_b_to_a}
    imgs = {'A': A, 'B': B}
    warped_frames = {'A': [], 'B': []}

    # find the difference between the identity and the mapping
    diff_maps = {}
    for k in mappings.keys():
        diff_maps[k] = mappings[k] - idty_map

    # quantum the mappings to n_segments, and create the video by warping the images with quantum mappings
    for n in range(n_segments):
        for k in mappings.keys():
            quantum_maps = idty_map + quantize_map(diff_maps[k], n_segments, n)
            warped_frames[k].append(warp_image(imgs[k], quantum_maps))
    mixer = partial(mix_frames_and_write, writer=writer, n_segments=n_segments, level=level, save_dir=save_dir)

    # write video
    warped_frames['B'].reverse()
    mixer(direction='a_to_b', warped_frames=warped_frames)

    if do_twosided:
        warped_frames['B'].reverse()
        warped_frames['A'].reverse()
        mixer(direction='b_to_a', warped_frames=warped_frames)

    writer.close()
    return


def find_middle_mapping(mappings, MF):
    idty_map = MF.identity_map(mappings['A'].size()).float()
    diffs_a_to_b = mappings['A'] - idty_map
    diffs_b_to_a = mappings['B'] - idty_map
    mid_map = (diffs_a_to_b.abs() + diffs_b_to_a.abs()) / 2
    return {'A': idty_map + mid_map, 'B': idty_map - mid_map}
