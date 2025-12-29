#!/usr/bin/env python3
"""
make_holograms_enhanced.py

Enhanced hologram generator with:
 - Angular Spectrum and Fresnel propagation (batch GPU processing)
 - Optional reference plane wave (inline holography): U_total = U_ref + U_obj
 - Inference mode: load a UNet checkpoint and reconstruct generated holograms, save reconstructions and compute PSNR
 - Progress bars, batching, and flexible I/O (.png/.npy)

Usage examples:
1) Generate holograms with reference wave and batching on GPU:
python make_holograms_enhanced.py --data_dir /path/to/data \
    --lambda_um 0.532 --z_mm 1.0 --dx_um 1.12 --model angular \
    --pad_factor 1.5 --device cuda --batch_size 8 --use_reference --ref_amp 1.0 --ref_phase 0.0

2) Generate holograms and run UNet inference (providing path to model checkpoint):
python make_holograms_enhanced.py --data_dir /path/to/data \
    --lambda_um 0.532 --z_mm 1.0 --dx_um 1.12 --model angular \
    --infer_model /path/to/best.pth --save_recon_ext .png

Notes:
 - Ground-truth amplitude images expected in data_dir/gt/
 - Output holograms go to data_dir/holo/ and reconstructions to data_dir/recon/
 - The script contains a small UNet definition compatible with the one I provided earlier. It will try to load the checkpoint provided by --infer_model.

Requirements:
 pip install torch torchvision numpy pillow tqdm

"""

import os
import argparse
from pathlib import Path
from glob import glob
import csv

import numpy as np
from PIL import Image

import torch
import torch.fft as fft
import torch.nn as nn
from tqdm import tqdm


def fixhololevel(pdata):
    data = None
    # grayscale only
    if len(pdata.shape) == 3 and pdata.shape[2] == 3:
        data = np.float32((pdata[...,1]).to('cpu').numpy())
        # data = cp.float32(cv2.cvtColor(pdata,cv2.COLOR_BGR2GRAY))
    else:
        if type(pdata) is torch.Tensor:
            data=pdata.cpu().numpy()
        else:
            data = np.float32(pdata)
    data = np.abs(data) 
    # normalize to 0..1
    data -= data.min()
    data /= data.max()
    # compute histogram
    datahst = np.histogram(data, bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(mostcommon)
    data = data ** n

    return data



# -------------------------------
# Lightweight UNet (for inference)
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batchnorm),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not batchnorm),
                  nn.ReLU(inplace=True)]
        if batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_ch))
            layers.insert(4, nn.BatchNorm2d(out_ch))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[32,64,128,256], bilinear=True):
        super().__init__()
        self.inc = ConvBlock(in_ch, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features)-1):
            self.downs.append(Down(features[i], features[i+1]))
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        rev_features = list(reversed(features))
        self.ups = nn.ModuleList()
        in_ch_ = features[-1]*2
        for f in rev_features:
            self.ups.append(Up(in_ch_, f, bilinear=bilinear))
            in_ch_ = f
        self.out_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
    def forward(self, x):
        encs = []
        x = self.inc(x)
        encs.append(x)
        for d in self.downs:
            x = d(x)
            encs.append(x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.ups):
            x = up(x, encs[-(i+2)])
        x = self.out_conv(x)
        return x

# -------------------------------
# Utilities
# -------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")

def load_image_as_float(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    else:
        img = Image.open(str(path)).convert("L")
        arr = np.array(img, dtype=np.float32)
    # normalize to [0,1]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / (arr.max() + 1e-12)
    return arr.astype(np.float32)


def save_float_image_uint8(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    im8 = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(im8).save(str(path))


def pad_to_factor(arr, factor):
    if factor <= 1.0:
        return arr, (0,0,0,0)
    h, w = arr.shape
    new_h = int(np.ceil(h * factor))
    new_w = int(np.ceil(w * factor))
    pad_h = new_h - h
    pad_w = new_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = np.pad(arr, ((top,bottom),(left,right)), mode='constant', constant_values=0.0)
    return padded, (top,bottom,left,right)


def unpad_array(arr, pad):
    top, bottom, left, right = pad
    h, w = arr.shape[-2], arr.shape[-1]
    return arr[..., top : h - bottom, left : w - right]


def make_frequency_grid(nx, ny, dx, device):
    fx = torch.fft.fftfreq(nx, d=dx, device=device)
    fy = torch.fft.fftfreq(ny, d=dx, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    return FX, FY


def angular_spectrum_propagation_batch(u0_batch, wavelength, z, dx, device, evanescent_filter=True):
    # u0_batch: [B, H, W] complex
    B, H, W = u0_batch.shape
    FX, FY = make_frequency_grid(W, H, dx, device)
    k = 2.0 * np.pi / wavelength
    lambda_fx2 = (wavelength * FX) ** 2
    lambda_fy2 = (wavelength * FY) ** 2
    arg = 1.0 - lambda_fx2 - lambda_fy2
    arg_c = arg.to(dtype=torch.complex64)
    sqrt_term = torch.sqrt(arg_c)
    H_transfer = torch.exp(1j * k * z * sqrt_term)
    if evanescent_filter:
        H_transfer = H_transfer * (arg >= 0).to(dtype=torch.complex64)
    # perform FFTs batch-wise
    U0 = fft.fft2(u0_batch)
    U1 = U0 * H_transfer
    u1 = fft.ifft2(U1)
    return u1


def fresnel_propagation_batch(u0_batch, wavelength, z, dx, device):
    B, H, W = u0_batch.shape
    k = 2.0 * np.pi / wavelength
    fx = torch.fft.fftfreq(W, d=dx, device=device)
    fy = torch.fft.fftfreq(H, d=dx, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    H_transfer = torch.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2)).to(dtype=torch.complex64)
    U0 = fft.fft2(u0_batch)
    U1 = U0 * H_transfer
    u1 = fft.ifft2(U1)
    return u1


def psnr(a, b, data_range=1.0):
    mse = torch.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * torch.log10(torch.tensor(data_range)) - 10.0 * torch.log10(mse)

# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--lambda_um', type=float, default=0.532)
    p.add_argument('--z_mm', type=float, default=1.0)
    p.add_argument('--dx_um', type=float, default=1.12)
    p.add_argument('--model', choices=('angular','fresnel'), default='fresnel')
    p.add_argument('--pad_factor', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--use_reference', action='store_true')
    p.add_argument('--ref_amp', type=float, default=1.0)
    p.add_argument('--ref_phase', type=float, default=0.0)
    p.add_argument('--infer_model', type=str, default='')
    p.add_argument('--save_recon_ext', type=str, default='.png')
    p.add_argument('--out_ext', type=str, default='.png')
    p.add_argument('--evanescent', action='store_true')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    gt_dir = data_dir / 'gt'
    holo_dir = data_dir / 'holo'
    recon_dir = data_dir / 'recon'
    os.makedirs(holo_dir, exist_ok=True)
    if args.infer_model:
        os.makedirs(recon_dir, exist_ok=True)

    wavelength = float(args.lambda_um) * 1e-6
    z = float(args.z_mm) * 1e-3
    dx = float(args.dx_um) * 1e-6
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' else 'cpu')

    # collect files
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted(glob(str(gt_dir / f'*{ext}'))))
    if len(files) == 0:
        raise RuntimeError(f'No ground-truth files in {gt_dir}')

    # batching
    batch_size = max(1, args.batch_size)
    # optional UNet load if inference requested
    unet = None
    if args.infer_model:
        ckpt_path = Path(args.infer_model)
        if not ckpt_path.exists():
            raise RuntimeError(f'infer_model checkpoint not found: {ckpt_path}')
        # instantiate model (assume in/out channels =1)
        unet = UNet(in_ch=1, out_ch=1).to(device)
        state = torch.load(str(ckpt_path), map_location=device)
        # support both state dict and full checkpoint
        if 'model_state' in state and isinstance(state['model_state'], dict):
            sd = state['model_state']
        elif 'model_state_dict' in state:
            sd = state['model_state_dict']
        else:
            sd = state
        unet.load_state_dict(sd)
        unet.eval()

    # process in batches
    n = len(files)
    out_metrics = []
    for i in tqdm(range(0, n, batch_size), desc='Batches'):
        batch_files = files[i : i + batch_size]
        amps = []
        pads = []
        names = []
        for f in batch_files:
            a = load_image_as_float(f)
            padded, pad = pad_to_factor(a, args.pad_factor)
            amps.append(padded)
            pads.append(pad)
            names.append(Path(f).name)
        # stack into tensor [B,H,W]
        amp_t = torch.from_numpy(np.stack(amps, axis=0)).to(device=device, dtype=torch.float32)
        # form complex field: amplitude * exp(i*phase=0)
        u0 = amp_t.to(dtype=torch.complex64)
        # optionally add reference plane wave (same shape) -> U_ref = A_ref * exp(i*phi)
        if args.use_reference:
            # create reference complex scalar and broadcast
            ref = args.ref_amp * np.exp(1j * args.ref_phase)
            # create tensor
            Uref = torch.zeros_like(u0, dtype=torch.complex64)
            Uref.real = args.ref_amp
            Uref.imag = 0.0
            # if phase not zero, rotate
            if abs(args.ref_phase) > 1e-12:
                ph = torch.tensor(np.cos(args.ref_phase), device=device, dtype=torch.float32) + 1j * torch.tensor(np.sin(args.ref_phase), device=device, dtype=torch.float32)
                Uref = Uref * ph
            u0_total = u0 + Uref
        else:
            u0_total = u0

        # propagate batch
        if args.model == 'angular':
            u1 = angular_spectrum_propagation_batch(u0_total, wavelength, z, dx, device, evanescent_filter=args.evanescent)
        else:
            u1 = fresnel_propagation_batch(u0_total, wavelength, z, dx, device)

        inten = (u1.abs() ** 2).to(dtype=torch.float32)
        inten = fixhololevel(inten)
        # unpad and save each
        for bi, name in enumerate(names):
            pad = pads[bi]
            inten_i = inten[bi]
            if args.pad_factor > 1.0:
                inten_i = unpad_array(inten_i, pad)
            inten_np = inten_i.copy()
            inten_np = inten_np - inten_np.min()
            if inten_np.max() > 0:
                inten_np = inten_np / (inten_np.max() + 1e-12)
            inten_np = fixhololevel(inten_np)
            out_name = Path(name).stem + args.out_ext
            out_path = holo_dir / out_name
            if args.out_ext.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                save_float_image_uint8(inten_np, out_path)
            else:
                np.save(str(out_path), inten_np.astype(np.float32))

        # if inference requested: run UNet on generated holograms (batch)
        if unet is not None:
            # prepare input to UNet: single-channel normalized intensity -> tensor [B,1,H,W]
            in_batch = inten.unsqueeze(1)  # complex? inten is real float
            # convert to float tensor
            in_batch = in_batch.to(dtype=torch.float32)
            with torch.no_grad():
                recon_batch = unet(in_batch)
            # clip/normalize per-image and save, compute PSNR vs GT
            for bi, name in enumerate(names):
                recon = recon_batch[bi,0].detach().cpu()
                # normalize to 0-1 per-image
                recon = recon - recon.min()
                if recon.max() > 0:
                    recon = recon / (recon.max() + 1e-12)
                recon_np = recon.numpy()
                out_rname = Path(name).stem + args.save_recon_ext
                out_rpath = recon_dir / out_rname
                if args.save_recon_ext.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                    save_float_image_uint8(recon_np, out_rpath)
                else:
                    np.save(str(out_rpath), recon_np.astype(np.float32))
                # compute PSNR vs GT (original GT image)
                gt = load_image_as_float(gt_dir / name)
                gt_t = torch.from_numpy(gt).to(dtype=torch.float32)
                # ensure shapes match (if pad_factor or resizing differs, user must ensure same dimensions)
                min_h = min(gt_t.shape[0], recon.shape[0])
                min_w = min(gt_t.shape[1], recon.shape[1])
                gt_crop = gt_t[:min_h, :min_w]
                recon_crop = recon[:min_h, :min_w]
                val_psnr = float(psnr(gt_crop, recon_crop, data_range=1.0).cpu().numpy())
                out_metrics.append({'file': name, 'psnr': val_psnr})

    # write metrics CSV if any
    if len(out_metrics) > 0:
        csv_path = data_dir / 'recon_metrics.csv'
        with open(csv_path, 'w', newline='') as cf:
            w = csv.DictWriter(cf, fieldnames=['file','psnr'])
            w.writeheader()
            for r in out_metrics:
                w.writerow(r)
        print('Wrote recon metrics to', csv_path)

    print('Done. Holograms in', holo_dir, ('; reconstructions in ' + str(recon_dir)) if args.infer_model else '')

if __name__ == '__main__':
    main()
