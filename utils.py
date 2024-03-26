import argparse
import os
import os.path as osp
import random
from enum import Enum

import numpy as np
import rawpy
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Lightweight LLIE')
    parser.add_argument(
        '--cfa',
        type=str,
        choices=['bayer', 'xtrans'],
        default='bayer'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./dataset/SID/Sony',
        help='Path to dataset',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='result',
        help='path to save checkpoint and log',
    )
    parser.add_argument(
        '--num_epoch', type=int, default=4000, help='number of training epochs'
    )
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size')
    parser.add_argument(
        '--patch_size', type=int, default=512, help='size of image patches for training'
    )
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument(
        '--milestones',
        nargs='+',
        type=int,
        default=[2000],
        help='when to reduce learning rate',
    )
    parser.add_argument(
        '--workers', type=int, default=4, help='number of data loading workers'
    )
    parser.add_argument(
        '--seed', type=int, default=-1, help='random seed (-1 for no manual seed)'
    )
    parser.add_argument('--size_divisibility', type=int, default=32)
    parser.add_argument(
        '--memorize',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='save data to memory',
    )
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--eval_step', type=int, default=200)
    parser.add_argument('--eval_begin_from', type=int, default=1600)
    parser.add_argument('--save_step', type=int, default=200)
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--ckpt', type=str, default=None, help='path to checkpoint')

    return init_args(parser.parse_args())


def init_args(args):
    if args.phase == 'test' and args.ckpt is None:
        assert 'checkpoint should be specified in the test phase'

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'checkpoints'), exist_ok=True)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return args


def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns


def compute_expo_ratio(input_fn, target_fn):
    in_exposure = float(osp.basename(input_fn)[9:-5])
    gt_exposure = float(osp.basename(target_fn)[9:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio


def pack_raw_bayer(raw):
    # pack Bayer image to 4 channels
    img = raw.raw_image_visible.copy().astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    white_point = raw.white_level
    H, W = img.shape[:2]

    out = np.stack(
        (
            img[R[0][0]: H: 2, R[1][0]: W: 2],  # RGBG
            img[G1[0][0]: H: 2, G1[1][0]: W: 2],
            img[B[0][0]: H: 2, B[1][0]: W: 2],
            img[G2[0][0]: H: 2, G2[1][0]: W: 2],
        ),
        axis=0,
    ).astype(np.float32)

    black_level = np.float32(raw.black_level_per_channel)[..., np.newaxis, np.newaxis]

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    return out


def pack_raw_xtrans(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = (im - 1024) / (16383 - 1024)  # subtract the black level
    im = np.clip(im, 0, 1)

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((9, H // 3, W // 3), dtype=np.float32)
    # 0 R
    out[0, 0::2, 0::2] = im[0:H:6, 0:W:6]
    out[0, 0::2, 1::2] = im[0:H:6, 4:W:6]
    out[0, 1::2, 0::2] = im[3:H:6, 1:W:6]
    out[0, 1::2, 1::2] = im[3:H:6, 3:W:6]

    # 1 G
    out[1, 0::2, 0::2] = im[0:H:6, 2:W:6]
    out[1, 0::2, 1::2] = im[0:H:6, 5:W:6]
    out[1, 1::2, 0::2] = im[3:H:6, 2:W:6]
    out[1, 1::2, 1::2] = im[3:H:6, 5:W:6]

    # 1 B
    out[2, 0::2, 0::2] = im[0:H:6, 1:W:6]
    out[2, 0::2, 1::2] = im[0:H:6, 3:W:6]
    out[2, 1::2, 0::2] = im[3:H:6, 0:W:6]
    out[2, 1::2, 1::2] = im[3:H:6, 4:W:6]

    # 4 R
    out[3, 0::2, 0::2] = im[1:H:6, 2:W:6]
    out[3, 0::2, 1::2] = im[2:H:6, 5:W:6]
    out[3, 1::2, 0::2] = im[5:H:6, 2:W:6]
    out[3, 1::2, 1::2] = im[4:H:6, 5:W:6]

    # 5 B
    out[4, 0::2, 0::2] = im[2:H:6, 2:W:6]
    out[4, 0::2, 1::2] = im[1:H:6, 5:W:6]
    out[4, 1::2, 0::2] = im[4:H:6, 2:W:6]
    out[4, 1::2, 1::2] = im[5:H:6, 5:W:6]

    out[5, :, :] = im[1:H:3, 0:W:3]
    out[6, :, :] = im[1:H:3, 1:W:3]
    out[7, :, :] = im[2:H:3, 0:W:3]
    out[8, :, :] = im[2:H:3, 1:W:3]
    return out


class SIDDataset(Dataset):
    def __init__(
        self,
        data_path,
        paired_fns,
        cfa,
        augment=True,
        repeat=1,
        patch_size=512,
        memorize=True,
        size_divisibility=32,
    ):
        super(SIDDataset, self).__init__()

        self.data_dir = data_path
        self.paired_fns = read_paired_fns(paired_fns)
        self.augment = augment
        self.patch_size = patch_size
        self.repeat = repeat
        self.size_divisibility = size_divisibility

        if cfa == 'xtrans':
            self.pack_raw = pack_raw_xtrans
            self.cfa_size = 3
        elif cfa == 'bayer':
            self.pack_raw = pack_raw_bayer
            self.cfa_size = 2
        else:
            raise NotImplementedError

        self.memorize = memorize
        self.target_dict = {}
        self.input_dict = {}

        if memorize:
            for i in tqdm(
                range(len(self)), desc='Loading dataset to memory', colour='cyan'
            ):
                _ = self[i]

    def __getitem__(self, i):
        i = i % len(self.paired_fns)
        input_fn, target_fn = self.paired_fns[i]

        input_path = osp.join(self.data_dir, 'short', input_fn)
        target_path = osp.join(self.data_dir, 'long', target_fn)

        if self.memorize:
            if target_fn not in self.target_dict:
                with rawpy.imread(target_path) as raw_target:
                    target_image = raw_target.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=True,
                        output_bps=16,
                    )
                    target_image = np.float32(target_image / 65535.0)
                    target_image = np.transpose(target_image, (2, 0, 1))
                    self.target_dict[target_fn] = target_image

            if input_fn not in self.input_dict:
                with rawpy.imread(input_path) as raw_input:
                    ratio = compute_expo_ratio(input_fn, target_fn)
                    input_image = self.pack_raw(raw_input) * ratio
                    self.input_dict[input_fn] = input_image

            input_image = self.input_dict[input_fn]
            target_image = self.target_dict[target_fn]
        else:
            with rawpy.imread(target_path) as raw_target:
                target_image = raw_target.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=16,
                )
                target_image = np.float32(target_image / 65535.0)
                target_image = np.transpose(target_image, (2, 0, 1))

            with rawpy.imread(input_path) as raw_input:
                ratio = compute_expo_ratio(input_fn, target_fn)
                input_image = self.pack_raw(raw_input) * ratio

        if self.augment:
            H, W = input_image.shape[-2:]
            ps = self.patch_size

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            input = input_image[:, yy: yy + ps, xx: xx + ps]
            target = target_image[
                :,
                self.cfa_size * yy: self.cfa_size * (yy + ps),
                self.cfa_size * xx: self.cfa_size * (xx + ps),
            ]

            if np.random.randint(2, size=1)[0] == 1:  # horizontal flip
                input = np.flip(input, axis=1)
                target = np.flip(target, axis=1)
            if np.random.randint(2, size=1)[0] == 1:  # vertical flip
                input = np.flip(input, axis=2)
                target = np.flip(target, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input = np.transpose(input, (0, 2, 1))
                target = np.transpose(target, (0, 2, 1))
        else:
            H, W = input_image.shape[-2:]
            size_divisibility = self.size_divisibility

            new_h = ((H + size_divisibility) // size_divisibility) * size_divisibility
            new_w = ((W + size_divisibility) // size_divisibility) * size_divisibility
            pad_h = new_h - H if H % size_divisibility != 0 else 0
            pad_w = new_w - W if W % size_divisibility != 0 else 0
            input = np.pad(input_image, ((0, 0), (0, pad_h), (0, pad_w)), 'reflect')
            target = target_image

        input = np.clip(input, 0.0, 1.0)
        target = target.copy()

        dic = {'input': input, 'target': target, 'fn': osp.splitext(input_fn)[0]}
        return dic

    def __len__(self):
        return len(self.paired_fns) * self.repeat


class AverageMeter(object):
    """Computes and stores the average and current value"""

    class Summary(Enum):
        NONE = 0
        AVERAGE = 1
        SUM = 2
        COUNT = 3

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt_str.format(**self.__dict__)

    def summary(self):
        fmt_str = ''
        if self.summary_type is self.Summary.NONE:
            fmt_str = ''
        elif self.summary_type is self.Summary.AVERAGE:
            fmt_str = '{name} {avg:.3f}'
        elif self.summary_type is self.Summary.SUM:
            fmt_str = '{name} {sum:.3f}'
        elif self.summary_type is self.Summary.COUNT:
            fmt_str = '{name} {count:.3f}'
        else:
            raise ValueError(f'invalid summary type {self.summary_type}')

        return fmt_str.format(**self.__dict__)
