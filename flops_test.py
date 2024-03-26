'''
1. for testing cpu inference time: python flops_test.py --cpu
2. for testing gpu inference time: python flops_test.py
3. for testing gpu memory consumption: python flops_test.py --no_benchmark
'''

import argparse
import os
import time

import numpy as np
import rawpy
import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from tqdm import tqdm

from model import LLIE

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./dataset/SID/Sony', help='Path to dataset')
parser.add_argument('--cfa', type=str, choices=['bayer', 'xtrans'], default='bayer')
parser.add_argument('--repeat', type=int, default=50, help='number of repeats')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--no_benchmark', action='store_true', help='no cudnn benchmarking')
parser.add_argument('--size_divisibility', type=int, default=32)
parser.add_argument('--no_pack', action='store_true')
args = parser.parse_args()

if torch.cuda.is_available() and not args.cpu:
    device = torch.device('cuda')
    use_gpu = True
    print('Using GPU: ', torch.cuda.get_device_name(0))
    if not args.no_benchmark:
        torch.backends.cudnn.benchmark = True  # use cudnn
        print('Using cudnn.benchmark')
else:
    device = torch.device('cpu')
    use_gpu = False
    print('Using CPU')

cfa_size = {
    'bayer': 2,
    'xtrans': 3
}[args.cfa]
net = LLIE(cfa_size=cfa_size).to(device)
net.eval()

# calculate macs and params
if args.no_pack:
    input_shape = {
        'bayer': (1, 1, 256, 256),
        'xtrans': (1, 1, 384, 384)
    }[args.cfa]
else:
    input_shape = {
        'bayer': (1, 4, 128, 128),  # <- [1, 1, 256, 256]
        'xtrans': (1, 9, 128, 128)  # <- [1, 1, 384, 384]
    }[args.cfa]

input = torch.randn(input_shape).to(device)
flops = FlopCountAnalysis(net, input)
with open(f'summary.txt', 'w', encoding='utf-8') as f:
    f.write(flop_count_table(flops))
    print('Flop count results written to summary.txt')

if args.cfa == 'bayer':
    path = os.path.join(args.data_path, 'long', '00001_00_10s.ARW')
    with rawpy.imread(path) as raw:
        arr = raw.raw_image_visible.copy()

    arr = arr.astype(np.float32)
    arr = np.maximum(arr - 512, 0) / (16383 - 512)
    arr = np.expand_dims(arr, axis=tuple(range(4 - arr.ndim)))
    tensor = torch.from_numpy(arr).to(device)
    if not args.no_pack:
        tensor = F.pixel_unshuffle(tensor, 2)  # simulate packing
else:
    path = os.path.join(args.data_path, 'long', '00001_00_10s.RAF')
    with rawpy.imread(path) as raw:
        arr = raw.raw_image_visible.copy()

    arr = arr.astype(np.float32)
    arr = np.maximum(arr - 1024, 0) / (16383 - 1024)
    arr = np.expand_dims(arr, axis=tuple(range(4 - arr.ndim)))

    H = (arr.shape[-1] // 6) * 6
    W = (arr.shape[-2] // 6) * 6
    arr = arr[..., :W, :H]

    tensor = torch.from_numpy(arr).to(device)
    if not args.no_pack:
        tensor = F.pixel_unshuffle(tensor, 3)  # simulate packing


if args.size_divisibility > 1:
    h, w = tensor.shape[-2:]
    new_h = ((h + args.size_divisibility) // args.size_divisibility) * args.size_divisibility
    new_w = ((w + args.size_divisibility) // args.size_divisibility) * args.size_divisibility
    pad_h = new_h - h if h % args.size_divisibility != 0 else 0
    pad_w = new_w - w if w % args.size_divisibility != 0 else 0
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), 'reflect')

total_time = 0
pbar = tqdm(range(args.repeat))

if not use_gpu:
    net = net.cpu()

if use_gpu:
    torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    if not args.no_benchmark:
        # dry run, let cudnn determine the best conv algorithm
        _ = net(tensor)

    for _ in pbar:
        if use_gpu:
            torch.cuda.synchronize()

        start_time = time.time()
        _ = net(tensor)

        if use_gpu:
            torch.cuda.synchronize()

        duration = time.time() - start_time
        total_time += duration
        pbar.set_description(f'{duration:.4f}')

print(f'Avg runtime: {(total_time/args.repeat):.4f}')

if use_gpu:
    print(f'Peak memory usage: {torch.cuda.max_memory_allocated(device)}')
