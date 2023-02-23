from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import LLIE
from PIL import Image
from pytorch_msssim import MS_SSIM
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter, SIDDataset, parse_args


def evaluate_model(net, data_loader, save_dir, num_epoch=None):
    net.eval()
    avg_psnr, avg_ssim = AverageMeter('PSNR'), AverageMeter('SSIM')

    with torch.inference_mode():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for data in test_bar:
            lq, gt, fn = (
                data['input'].to(device),
                data['target'].to(device),
                data['fn'][0],
            )

            out = net(lq).squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

            out = (out * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)

            h, w, _ = gt.shape
            out = out[:h, :w, :]

            current_psnr = compute_psnr(out, gt)
            current_ssim = compute_ssim(
                out,
                gt,
                channel_axis=2,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )

            avg_psnr.update(current_psnr)
            avg_ssim.update(current_ssim)

            save_path = Path(save_dir, 'val_result', str(num_epoch), f'{fn}.png')
            save_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(out).save(save_path)
            test_bar.set_description(
                f'Test Epoch: [{num_epoch}] '
                f'PSNR: {avg_psnr.avg:.2f} SSIM: {avg_ssim.avg:.4f}'
            )

    return avg_psnr.avg, avg_ssim.avg


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu') if args.cpu else torch.device('cuda')

    test_dataset = SIDDataset(
        args.data_path,
        'Sony_test.txt',
        augment=False,
        pad_multiple_to=args.pad_multiple_to,
        memorize=args.memorize,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    save_path = Path(args.save_path)

    model = LLIE().to(device)
    if args.phase == 'test':
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        evaluate_model(model, test_loader, save_path, 'final')
    else:
        msssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        optimizer = Adam(model.parameters(), lr=args.lr)
        lr_scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        total_loss, total_num = 0.0, 0
        train_dataset = SIDDataset(
            args.data_path,
            'Sony_train.txt',
            patch_size=args.patch_size,
            augment=True,
            memorize=args.memorize,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

        for n_epoch in range(1, args.num_epoch + 1):
            train_bar = tqdm(train_loader, dynamic_ncols=True)
            for data in train_bar:  # train
                model.train()
                lq, gt = data['input'].to(device), data['target'].to(device)
                out = model(lq)
                loss = F.l1_loss(out, gt) + 1.5 * (1 - msssim_loss(out, gt))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_num += lq.size(0)
                total_loss += loss.item() * lq.size(0)
                train_bar.set_description(
                    f'Train Epoch: [{n_epoch}/{args.num_epoch+1}] Loss: {total_loss / total_num:.3f}'
                )

            lr_scheduler.step()

            if (
                n_epoch >= args.eval_begin_from and n_epoch % args.eval_step == 0
            ):  # evaluate
                val_psnr, val_ssim = evaluate_model(
                    model, test_loader, save_path, n_epoch
                )

                # save statistics
                with save_path.joinpath('record.txt').open(mode='a+') as f:
                    f.write(
                        f'Epoch: {n_epoch} PSNR:{val_psnr:.2f} SSIM:{val_ssim:.4f}\n'
                    )

            if n_epoch % args.save_step == 0:
                torch.save(
                    model.state_dict(),
                    save_path.joinpath('checkpoints', f'{n_epoch}.pth'),
                )
