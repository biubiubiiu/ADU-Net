'''
1. for testing cpu inference time: python flops_test.py --model $MODEL --cpu
2. for testing gpu inference time: python flops_test.py --model $MODEL
3. for testing max gpu memory: python flops_test.py --model $MODEL --no_benchmark
'''

import argparse
import time

import numpy as np
import rawpy
import torch
from model import LLIE
from torchinfo import summary
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=50, help='number of repeats')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--no_benchmark', action='store_true', help='no cudnn benchmarking')
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

net = LLIE().to(device)
net.eval()

# calculate macs and params
input_shape = (1, 4, 128, 128)  # <- [1, 1, 256, 256]
s = summary(net, input_shape, col_names=['num_params', 'mult_adds'], depth=4, verbose=0)
with open(f'summary.txt', 'w', encoding='utf-8') as f:
    f.write(str(s))


path = '/data/SID/Sony/long/00001_00_10s.ARW'
with rawpy.imread(path) as raw:
    arr = raw.raw_image_visible.copy()

arr = arr.astype(np.float32)
arr = np.maximum(arr - 512, 0) / (16383 - 512)  # subtract the black level and normalize
arr = np.expand_dims(arr, axis=tuple(range(4 - arr.ndim)))
tensor = torch.from_numpy(arr).to(device)

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
    print(torch.cuda.max_memory_allocated(device))
