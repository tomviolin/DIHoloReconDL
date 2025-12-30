#!/usr/bin/env python3
"""
DIHM UNet
--------------
A self-contained PyTorch script providing:
 - a simple configurable U-Net (encoder-decoder) for single-channel hologram -> reconstruction
 - a Dataset class that expects paired hologram / ground-truth images (supports PNG, JPG, or .npy)
 - training and validation loop with checkpoints and basic logging
 - example usage at the bottom

Notes / assumptions:
 - Input and target are single-channel (grayscale). If your data is RGB, set `in_ch`/`out_ch` accordingly.
 - The dataset expects a directory with two subfolders: `holo/` and `gt/`, with matching filenames.
 - Uses L1 + MSE loss by default; replace/add perceptual/SSIM losses if desired.

Usage:
 python dihm_unet.py --data_dir /path/to/data --epochs 100 --batch_size 8

"""

import os, sys, math
from dialog import Dialog
import locale
locale.setlocale(locale.LC_ALL, '')
dlg = Dialog(dialog="dialog")
# sset up dialog to have no background
dlg.set_background_title("DIHM UNet Trainer")


import os, sys, math
import argparse
import json
from glob import glob
from pathlib import Path
from typing import List
import cupy  as cp
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchinfo import summary
import torch.nn.functional as F
import torchmetrics

trial_dir = None

import datetime
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
## establish trial number
trial_num = 1
past_trials = sorted(glob(f'trials/trial*'))
if len(past_trials) > 0:
    last_trial = os.path.basename(past_trials[-1])
    last_num = int(last_trial.replace('trial',''))
    trial_num = last_num + 1
trial_dir = f'trials/trial{trial_num:03d}'

dresult = dlg.inputbox(text=f'Starting trial {trial_num:03d}. Enter description:', title='New Trial', init='')
if dresult[0] != dlg.OK:
    print('Exiting...')
    sys.exit(0)

trial_desc = dresult[1].strip()
os.makedirs(trial_dir, exist_ok=True)
with open(os.path.join(trial_dir, 'trial.json'), 'w') as f:
    f.write(f'{{"trial_num": {trial_num}, "description": "{trial_desc}"}}\n')

print(f'Starting trial {trial_num:03d}: {trial_desc}')

# trap interrupts (Ctrl-C) to exit cleanly
def signal_handler(sig, frame):
    print('Exiting...')
    cv2.destroyAllWindows()
    sys.exit(0)

import signal
signal.signal(signal.SIGINT, signal_handler)


def guicoop():
    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        return 1
    else:
        return 0

def coop_exit():
    print('Exiting...')
    cv2.destroyAllWindows()
    sys.exit(0)




def fixhololevel(pdata):
    #print(f"fixhololevel: pdata shape = {pdata.shape}, dtype = {pdata.dtype}")
    if type(pdata) is torch.Tensor:
        data = pdata
    else:
        data = torch.from_numpy(pdata)
    data_intype = data.dtype
    data_indevice = data.device
    # normalize to 0..1
    data -= data.min()
    data /= data.max()
    # compute histogram
    datahst = cp.histogram(cp.asarray(data), bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = cp.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = cp.log(0.5) /  cp.log(mostcommon)
    data = cp.asarray(data) ** n

    return torch.as_tensor(data, device=data_indevice, dtype=data_intype)




def angular_spectrum_prop(u0, dx, wavelength, z):
    """
    Angular spectrum propagation of field u0 (2D complex numpy array)
    dx: sampling interval (same units as wavelength and z)
    wavelength: same units (um)
    z: propagation distance (um) — can be negative for backpropagation
    Returns propagated field u1 (2D complex numpy array)
    """
    # ensure complex
    u0 = np.asarray(u0, dtype=np.complex64)
    ny, nx = u0.shape
    k = 2 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * z * 2 * np.pi * np.sqrt(np.maximum(0.0, (1.0 / wavelength**2) - (FX**2 + FY**2))))
    # evanescent components handled by sqrt(max(0,...))
    U0 = np.fft.fft2(u0)
    U1 = U0 * H
    u1 = np.fft.ifft2(U1)
    return np.abs(u1)











def init_weights_to_random(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or \
         isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.BatchNorm2d):
        # Initialize biases to random numbers
        if m.bias is not None:
            if m.bias.data is not None:
                m.bias.data = torch.randn(m.bias.size())
        if m.weight is not None:
            if m.weight.data is not None:
                m.weight.data = torch.randn(m.weight.size()) * 0.01


class SamePaddingMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def forward(self, x):
        h_in, w_in = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        
        # Calculate padding needed for 'same'
        # Formula: pad = max((out_dim - 1) * stride + kernel_size - in_dim, 0)
        ph = max((np.ceil(h_in / sh) - 1) * sh + kh - h_in, 0)
        pw = max((np.ceil(w_in / sw) - 1) * sw + kw - w_in, 0)
        
        # Calculate top/bottom and left/right padding (asymmetric)
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
        
        # Pad with -inf for MaxPool (so padding doesn't affect the max value)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=float('-inf'))
        
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)


import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        # Block 1
        self.enc1_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_bn1 = nn.BatchNorm2d(64)
        self.enc1_relu1 = nn.ReLU(inplace=True)
        self.enc1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc1_bn2 = nn.BatchNorm2d(64)
        self.enc1_relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.enc2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_bn1 = nn.BatchNorm2d(128)
        self.enc2_relu1 = nn.ReLU(inplace=True)
        self.enc2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc2_bn2 = nn.BatchNorm2d(128)
        self.enc2_relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.enc3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_bn1 = nn.BatchNorm2d(256)
        self.enc3_relu1 = nn.ReLU(inplace=True)
        self.enc3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc3_bn2 = nn.BatchNorm2d(256)
        self.enc3_relu2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.enc4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_bn1 = nn.BatchNorm2d(512)
        self.enc4_relu1 = nn.ReLU(inplace=True)
        self.enc4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc4_bn2 = nn.BatchNorm2d(512)
        self.enc4_relu2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm2d(1024)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm2d(1024)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)
        
        # Decoder (Expanding Path)
        # Block 1
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec1_bn1 = nn.BatchNorm2d(512)
        self.dec1_relu1 = nn.ReLU(inplace=True)
        self.dec1_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec1_bn2 = nn.BatchNorm2d(512)
        self.dec1_relu2 = nn.ReLU(inplace=True)
        
        # Block 2
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec2_bn1 = nn.BatchNorm2d(256)
        self.dec2_relu1 = nn.ReLU(inplace=True)
        self.dec2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec2_bn2 = nn.BatchNorm2d(256)
        self.dec2_relu2 = nn.ReLU(inplace=True)
        
        # Block 3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec3_bn1 = nn.BatchNorm2d(128)
        self.dec3_relu1 = nn.ReLU(inplace=True)
        self.dec3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec3_bn2 = nn.BatchNorm2d(128)
        self.dec3_relu2 = nn.ReLU(inplace=True)
        
        # Block 4
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec4_bn1 = nn.BatchNorm2d(64)
        self.dec4_relu1 = nn.ReLU(inplace=True)
        self.dec4_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec4_bn2 = nn.BatchNorm2d(64)
        self.dec4_relu2 = nn.ReLU(inplace=True)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder Block 1
        enc1 = self.enc1_conv1(x)
        enc1 = self.enc1_bn1(enc1)
        enc1 = self.enc1_relu1(enc1)
        enc1 = self.enc1_conv2(enc1)
        enc1 = self.enc1_bn2(enc1)
        enc1 = self.enc1_relu2(enc1)
        pool1 = self.pool1(enc1)



        # Encoder Block 2
        enc2 = self.enc2_conv1(pool1)
        enc2 = self.enc2_bn1(enc2)
        enc2 = self.enc2_relu1(enc2)
        enc2 = self.enc2_conv2(enc2)
        enc2 = self.enc2_bn2(enc2)
        enc2 = self.enc2_relu2(enc2)
        pool2 = self.pool2(enc2)
        
        # Encoder Block 3
        enc3 = self.enc3_conv1(pool2)
        enc3 = self.enc3_bn1(enc3)
        enc3 = self.enc3_relu1(enc3)
        enc3 = self.enc3_conv2(enc3)
        enc3 = self.enc3_bn2(enc3)
        enc3 = self.enc3_relu2(enc3)
        pool3 = self.pool3(enc3)
        
        # Encoder Block 4
        enc4 = self.enc4_conv1(pool3)
        enc4 = self.enc4_bn1(enc4)
        enc4 = self.enc4_relu1(enc4)
        enc4 = self.enc4_conv2(enc4)
        enc4 = self.enc4_bn2(enc4)
        enc4 = self.enc4_relu2(enc4)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck_conv1(pool4)
        bottleneck = self.bottleneck_bn1(bottleneck)
        bottleneck = self.bottleneck_relu1(bottleneck)
        bottleneck = self.bottleneck_conv2(bottleneck)
        bottleneck = self.bottleneck_bn2(bottleneck)
        bottleneck = self.bottleneck_relu2(bottleneck)
        
        # Decoder Block 1
        up1 = self.upconv1(bottleneck)
        dec1 = torch.cat([up1, enc4], dim=1)
        dec1 = self.dec1_conv1(dec1)
        dec1 = self.dec1_bn1(dec1)
        dec1 = self.dec1_relu1(dec1)
        dec1 = self.dec1_conv2(dec1)
        dec1 = self.dec1_bn2(dec1)
        dec1 = self.dec1_relu2(dec1)
        
        # Decoder Block 2
        up2 = self.upconv2(dec1)
        dec2 = torch.cat([up2, enc3], dim=1)
        dec2 = self.dec2_conv1(dec2)
        dec2 = self.dec2_bn1(dec2)
        dec2 = self.dec2_relu1(dec2)
        dec2 = self.dec2_conv2(dec2)
        dec2 = self.dec2_bn2(dec2)
        dec2 = self.dec2_relu2(dec2)
        
        # Decoder Block 3
        up3 = self.upconv3(dec2)
        dec3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.dec3_conv1(dec3)
        dec3 = self.dec3_bn1(dec3)
        dec3 = self.dec3_relu1(dec3)
        dec3 = self.dec3_conv2(dec3)
        dec3 = self.dec3_bn2(dec3)
        dec3 = self.dec3_relu2(dec3)
        
        

        # Decoder Block 4
        up4 = self.upconv4(dec3)
        dec4 = torch.cat([up4, enc1], dim=1)
        dec4 = self.dec4_conv1(dec4)
        
        dec4 = self.dec4_bn1(dec4)
        dec4 = self.dec4_relu1(dec4)
        dec4 = self.dec4_conv2(dec4)
        dec4 = self.dec4_bn2(dec4)
        dec4 = self.dec4_relu2(dec4)
        
        # Output
        out = self.out_conv(dec4)
        
        return out

"""
# Example usage
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

"""


# -----------------------
# Data utilities
# -----------------------

IMG_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']

class PairedHoloDataset(Dataset):
    """Loads paired hologram -> ground-truth images
    Directory structure:
      data_dir/
        holo/
          sample1.png
          sample2.png
        gt/
          sample1.png
          sample2.png
    Filenames must match between holo/ and gt/.
    """
    def __init__(self, data_dir, split='train', transform=None, in_exts=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.holo_dir = self.data_dir / 'holo'
        self.gt_dir = self.data_dir / 'gt'
        self.transform = transform
        if in_exts is None:
            self.exts = IMG_EXTS
        else:
            self.exts = in_exts
        self.files = self._find_pairs()

    def _find_pairs(self) -> List[Path]:
        files = []
        for p in sorted(self.holo_dir.iterdir()):
            if p.suffix.lower() in self.exts:
                gt = self.gt_dir / p.name
                if gt.exists():
                    files.append(p.name)
        if len(files) == 0:
            raise RuntimeError(f'No paired files found in {self.holo_dir} and {self.gt_dir}')
        return files

    def __len__(self):
        return len(self.files)

    def _load(self, path: Path):
        p = Path(path)
        if p.suffix.lower() == '.npy':
            arr = np.load(str(p))
            if arr.ndim == 2:
                return arr.astype(np.float32)
            elif arr.ndim == 3:
                # assume HWC, convert to CHW
                return np.transpose(arr, (2,0,1)).astype(np.float32)
            else:
                raise RuntimeError('Unsupported .npy shape')
        else:
            img = Image.open(str(p)).convert('L')
            arr = np.array(img).astype(np.float32)
            #print(f"Loaded image {p} with shape {arr.shape} and min/max {arr.min()}/{arr.max()}")
            return arr

    def __getitem__(self, idx):
        name = self.files[idx]
        holo_p = self.holo_dir / name
        gt_p = self.gt_dir / name
        holo = self._load(holo_p)
        gt = self._load(gt_p)
        #holo = fixhololevel(holo)
        #gt = fixhololevel(gt)
        # normalize to [0,1]
        #holo = holo / 255.0
        #gt = gt / 255.0
        #holo = holo - holo.min()
        #if holo.max() > 0:
        #    holo = holo / holo.max()
        #gt = gt - gt.min()
        #if gt.max() > 0:
        #    gt = gt / gt.max()
        # ensure channel-first
        if holo.ndim == 2:
            holo = np.expand_dims(holo, 0)
        if gt.ndim == 2:
            gt = np.expand_dims(gt, 0)
        if self.transform is not None:
            return self.transform(torch.from_numpy(holo)), self.transform(torch.from_numpy(gt))
        return torch.from_numpy(holo), torch.from_numpy(gt)

class HoloDataset(Dataset):
    """Loads unpaired holograms. no ground-truth images.  
    Directory structure:
      data_dir/
        holo/
          sample1.png
          sample2.png
    """
    def __init__(self, data_dir, split=None, transform=None, in_exts=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.holo_dir = self.data_dir / 'holo'
        self.transform = transform
        if in_exts is None:
            self.exts = IMG_EXTS
        else:
            self.exts = in_exts
        self.files = self._find_files()

    def _find_files(self) -> List[Path]:
        files = []
        for p in sorted(self.holo_dir.iterdir()):
            files.append(p.name)
        if len(files) == 0:
            raise RuntimeError(f'No files found in {self.holo_dir}')
        return files

    def __len__(self):
        return len(self.files)

    def _load(self, path: Path):
        p = Path(path)
        if p.suffix.lower() == '.npy':
            arr = np.load(str(p))
            if arr.ndim == 2:
                return arr.astype(np.float32)
            elif arr.ndim == 3:
                # assume HWC, convert to CHW
                return np.transpose(arr, (2,0,1)).astype(np.float32)
            else:
                raise RuntimeError('Unsupported .npy shape')
        else:
            img = Image.open(str(p)).convert('L')
            arr = np.array(img).astype(np.float32)
            #print(f"Loaded image {p} with shape {arr.shape} and min/max {arr.min()}/{arr.max()}")
            return arr

    def __getitem__(self, idx):
        name = self.files[idx]
        holo_p = self.holo_dir / name
        holo = self._load(holo_p)
        # ensure channel-first
        if holo.ndim == 2:
            holo = np.expand_dims(holo, 0)
        if self.transform is not None:
            return self.transform(torch.from_numpy(holo))
        return torch.from_numpy(holo)


# -----------------------
# Training utilities
# -----------------------

def train_one_epoch(model, loader, optim, device, scaler, epoch):
    # how to change code so there's no mixed precision for gpu
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()


    # 
    model = model.float()
    model = model.to(device)
    model = model.train()
    #model = model.to(device)
    total_loss = 0.0
    loop_count = 0
    for x, y in loader:
        if args.blur_holo:
            k = 5 #np.random.randint(low=0,high=5, size=1)[0]*2+1
            sd = 1 #np.random.uniform(2,4,1)[0]
            xx = cv2.GaussianBlur(x[0,0].cpu().detach().numpy(), (k,k),sd)
            #xx += np.random.normal(0,0.07, xx.shape)
            x[0,0,...] = torch.from_numpy(xx)
        if args.blur_gt:
            k = 5 #np.random.randint(low=0,high=3, size=1)[0]*2+1
            sd = 1 #np.random.uniform(2,4,1)[0]
            yy = cv2.GaussianBlur(y[0,0].cpu().detach().numpy(), (k,k),sd)
            y[0,0,...] = torch.from_numpy(yy)
        loop_count += 1
        print(f"Train: {loop_count}/{len(loader)}", end='\r', flush=True)
        #print(f"Train one epoch: x shape = {x.shape}, y shape = {y.shape}")

        #print(F"x from loader shape: {x.shape}, y from loader shape: {y.shape}")
        x=fixhololevel(x)
        if args.z_mm != 0.0:
            newx = np.zeros((x.shape[0],3,x.shape[2],x.shape[3]), dtype=np.float32)
            newx[:,0,...] = x[:,0,...].cpu().detach().numpy()
            #print(F"x from loader foxhololeveled shape: {x.shape}, y from loader shape: {y.shape}")
            #x=fixhololevel(x)
            for b in range(x.shape[0]):
                holo_img = newx[b,0,...]
                #holo_img = (holo_img - holo_img.min()) / (holo_img.max() - holo_img.min())
                pic_in_prop1 = angular_spectrum_prop(holo_img, dx=args.dx_um, wavelength=args.lambda_um,
                                                 z=-args.z_mm*1000)
                pic_in_prop2= angular_spectrum_prop(holo_img, dx=args.dx_um, wavelength=args.lambda_um,
                                                 z=-args.z_mm*1000/2)


                newx[b,1,...] = torch.from_numpy(pic_in_prop1)
                newx[b,2,...] = torch.from_numpy(pic_in_prop2)
            x = fixhololevel(newx)
            #print(f"x after propagation shape: {x.shape}, y shape: {y.shape}")
        y = fixhololevel(y)
        x=x.to(dtype=torch.float).to(device=device)
        y=y.to(dtype=torch.float).to(device=device)
        if scaler is not None:
            #with torch.amp.autocast('cuda'):
            out = model(x)
            loss = loss_fn(out, y)
            #minloss = loss_fn(y,y)
            #print(f"  Loss: {loss.item():.6f}, MinLoss: {minloss.item():.6f}")
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            out = model(x)
            loss = loss_fn(out, y)
            #minloss = loss_fn(y,y)
            #print(f"  Loss: {loss.item():.6f}, MinLoss: {minloss.item():.6f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += float(loss.item()) * x.size(0)
        if guicoop():
            return True, total_loss / len(loader.dataset)
    return False, total_loss / len(loader.dataset)


def validate(model, loader, device, epoch):
    model.eval()
    #model.to(device)
    total_loss = 0.0
    datanum = 0
    with torch.no_grad():
        for x, y in loader:
            ##print(f"Validate: x shape = {x.shape}, y shape = {y.shape}")
            if guicoop(): coop_exit()
            #print(F"x from loader shape: {x.shape}, y from loader shape: {y.shape}")
            x=fixhololevel(x)
            if args.z_mm != 0.0:
                newx = np.zeros((x.shape[0],3,x.shape[2],x.shape[3]), dtype=np.float32)
                newx[:,0,...] = x[:,0,...].cpu().detach().numpy()
                #print(F"x from loader foxhololeveled shape: {x.shape}, y from loader shape: {y.shape}")
                #x=fixhololevel(x)
                for b in range(x.shape[0]):
                    holo_img = newx[b,0,...]
                    #holo_img = (holo_img - holo_img.min()) / (holo_img.max() - holo_img.min())
                    pic_in_prop1 = angular_spectrum_prop(holo_img, dx=args.dx_um, wavelength=args.lambda_um,
                                                     z=-args.z_mm*1000)
                    pic_in_prop2= angular_spectrum_prop(holo_img, dx=args.dx_um, wavelength=args.lambda_um,
                                                     z=-args.z_mm*1000/2)


                    newx[b,1,...] = torch.from_numpy(pic_in_prop1)
                    newx[b,2,...] = torch.from_numpy(pic_in_prop2)
                x = fixhololevel(newx)
                #print(f"x after propagation shape: {x.shape}, y shape: {y.shape}")
            y=fixhololevel(y)
            #yy = cv2.GaussianBlur(y[0,0].cpu().detach().numpy(), (3,3),1)
            #y[0,0,...] = torch.from_numpy(yy)
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += float(loss.item()) * x.size(0)
            """
            out_img = out[0,0].cpu().numpy()
            out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
            x_img = x[0,0].cpu().numpy()
            x_img = (x_img - x_img.min()) / (x_img.max() - x_img.min())
            y_img = y[0,0].cpu().numpy()
            y_img = (y_img - y_img.min()) / (y_img.max() - y_img.min())
            """


            #cv2.imshow('gt',  y  [0,0,...].cpu().detach().numpy()*255)
            pic_gt = np.clip((y[0,0].cpu().detach().numpy()*255),0,255).astype(np.uint8)
            pic_in0 = np.clip((x  [0,0,...].cpu().detach().numpy()*255),0,255).astype(np.uint8)
            if args.z_mm != 0.0:
                pic_in1 = np.clip((x  [0,1,...].cpu().detach().numpy()*255),0,255).astype(np.uint8)
                pic_in2 = np.clip((x  [0,2,...].cpu().detach().numpy()*255),0,255).astype(np.uint8)
            pic_out= np.clip((out[0,0,...].cpu().detach().numpy()*255),0,255).astype(np.uint8)



            """
            # def angular_spectrum_prop(u0, dx, wavelength, z):

            pic_in_prop = angular_spectrum_prop(x[0,0,...].cpu().detach().numpy(), dx=args.dx_um, wavelength=args.lambda_um,
                                                 z=-args.z_mm*1000)
            pic_in_prop = fixhololevel(pic_in_prop).cpu().detach().numpy()
            pic_in_prop = np.clip((pic_in_prop*255),0,255).astype(np.uint8)
            """
            divider = np.zeros((pic_gt.shape[0],10), dtype=np.uint8)
            if args.z_mm != 0.0:
                pic = np.hstack((pic_gt, divider,pic_in0,divider,pic_in1,divider,pic_in2, divider, pic_out)) #, divider, pic_in_prop))
            else:
                pic = np.hstack((pic_gt, divider,pic_in0,divider,pic_out)) #, divider, pic_in_prop))
            cv2.imshow('validation', pic)
            picdir = f'{trial_dir}/val_images'
            nowstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(picdir, exist_ok=True)
            datanum += 1
            cv2.imwrite(f'{picdir}/val_{epoch:03d}_{datanum:04d}_{nowstamp}.png', pic)

            #cv2.imshow('gt',  y[0,0].cpu().detach().numpy())
            #cv2.imshow('in',  x  [0,0,...].cpu().detach().numpy())
            #cv2.imshow('out', out[0,0,...].cpu().detach().numpy())
            #cv2.moveWindow('gt',  0,0)
            #cv2.moveWindow('in',  360,0)
            #cv2.moveWindow('out', 720,0)
            #print(f"gt range: {y.min().item():.4f} - {y.max().item():.4f}, in range: {x.min().item():.4f} - {x.max().item():.4f}, out range: {out.min().item():.4f} - {out.max().item():.4f}")
    return total_loss / len(loader.dataset)




def evaluate(model, loader, device, epoch):
    model.eval()
    model.to(device)
    datanum = 0
    with torch.no_grad():
        for x in loader:
            ##print(f"Validate: x shape = {x.shape}, y shape = {y.shape}")
            if guicoop(): coop_exit()
            #print(F"x from loader shape: {x.shape}, y from loader shape: {y.shape}")
            x=fixhololevel(x)
            x = x.to(device=device, dtype=torch.float)
            out = model(x)
            pic_in= np.clip((x  [0,0,...].cpu().detach().numpy()*255),0,255).astype(np.uint8)
            pic_out= np.clip((out[0,0,...].cpu().detach().numpy()*255),0,255).astype(np.uint8)

            divider = np.zeros((pic_in.shape[0],10), dtype=np.uint8)
            pic = np.hstack((pic_in,divider,pic_out)) #, divider, pic_in_prop))
            cv2.imshow('validation', pic)
            picdir = f'{trial_dir}/val_images'
            nowstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(picdir, exist_ok=True)
            datanum += 1
            cv2.imwrite(f'{picdir}/val_{epoch:03d}_{datanum:04d}_{nowstamp}.png', pic)

            #cv2.imshow('gt',  y[0,0].cpu().detach().numpy())
            #cv2.imshow('in',  x  [0,0,...].cpu().detach().numpy())
            #cv2.imshow('out', out[0,0,...].cpu().detach().numpy())
            #cv2.moveWindow('gt',  0,0)
            #cv2.moveWindow('in',  360,0)
            #cv2.moveWindow('out', 720,0)
            #print(f"gt range: {y.min().item():.4f} - {y.max().item():.4f}, in range: {x.min().item():.4f} - {x.max().item():.4f}, out range: {out.min().item():.4f} - {out.max().item():.4f}")
    return



def save_checkpoint(state, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, os.path.join(out_dir, f'best_{state["epoch"]:04d}.pth'))

# Define loss here (L1 + MSE)
l1 = nn.L1Loss()
mse = nn.MSELoss()

from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda' if torch.cuda.is_available() else 'cpu')

def ssim_loss_fn (pred,target):
    return 1.0-ssim_loss(pred, target)

loss_fn = mse
#loss_fn = nn.SmoothL1Loss(beta=0.5)

"""

# loss function based on torchmetrics.image StructuralSimilarityIndexMeasure
def loss_fn(pred, target):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    #return mse(pred, target)*pred.shape[0]*pred.shape[1] + l1(pred, target)*pred.shape[0]*pred.shape[1]/100
    return (1.0-ssim(pred, target))*1.9  + 2.2*l1(pred, target) + 2.2*mse(pred, target)
    
    C1 = 0.01 ** 2.0
    C2 = 0.03 ** 2.0
    mu_x = nn.functional.avg_pool2d(pred, 3, 1, 1)
    mu_y = nn.functional.avg_pool2d(target, 3, 1, 1)
    sigma_x = nn.functional.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x  
    sigma_y = nn.functional.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = nn.functional.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()
"""



    #return alpha * l1(pred, target) + (1-alpha) * mse(pred, target)

# -----------------------
# CLI / main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, help='root data dir containing holo/ and gt/')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--checkpoints', type=str, default=f'trials/trial{trial_num:03d}/chkpts', help='directory to save checkpoints')
    p.add_argument('--img_size', type=int, default=128, help='resize shorter edge to this and center-crop (keeps square)')
    p.add_argument('--in_ch', type=int, default=1)
    p.add_argument('--out_ch', type=int, default=1)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--blur_gt', action='store_true', help='if set, apply slight Gaussian blur to ground-truth images')
    p.add_argument('--blur_holo', action='store_true', help='if set, apply slight Gaussian blur to holo images')
    p.add_argument('--test_eval', action='store_true', help='if set, run evaluation only on test set (not training)')
    p.add_argument('--eval_only', action='store_true', help='if set, this will only eval, no gts are needed')
    p.add_argument('--load_checkpoint', type=str, default='', help='if set, load from checkpoint pth file')
    p.add_argument('--dx_um', type=float, default=1.2, help='pixel size in microns')
    p.add_argument('--lambda_um', type=float, default=0.650, help='wavelength in microns')
    p.add_argument('--z_mm', type=float, default=0.0, help='propagation distance in microns')
    return p.parse_args()

def make_transforms(img_size):
    # simple transforms: convert to float tensor, resize, center crop
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.CenterCrop(int(img_size)),
    ])

def main():
    global args, dlg, trial_dir, trial_num
    args = parse_args()
    argdict = vars(args)
    arglist = list(argdict)
    argtypes = [ type(argdict[k]) for k in arglist ]
    print("--- argdict ---")
    print(argdict)
    print("---------------")
    print("--- arglist ---")
    print(arglist)
    print("---------------")
    print("--- arg types ---")
    print(argtypes)
    print("------------------")

    # load dihm_unet.json
    json_config_file = 'dihm_unet.json'
    default_json_config_file = 'default_dihm_unet.json'
    if not os.path.exists(json_config_file):
        # copy default 
        if os.path.exists(default_json_config_file):
            import shutil
            shutil.copyfile(default_json_config_file, json_config_file)

    if os.path.exists(json_config_file):
        with open(json_config_file, 'r') as f:
            j = json.load(f)
            for k in j:
                if k in argdict:
                    argdict[k] = j[k]
    args.checkpoints = os.path.join(trial_dir, 'chkpts') 
    past_checkpoints = []
    for d in sorted(glob("trials*/trial*/*/*.pth")):
        mtime = os.path.getmtime(d)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        past_checkpoints.append( f"{mtime_str},{d}" )

    if len(past_checkpoints) > 0:
        for pc in sorted(past_checkpoints):
            pc_trial_dir = os.path.dirname( os.path.dirname(pc.split(',')[1]) )
            pc_trial_jsonfile = os.path.join(pc_trial_dir, 'trial.json')
            if os.path.exists(pc_trial_jsonfile):
                with open(pc_trial_jsonfile, 'r') as f:
                    try:
                        j = json.load(f)
                        desc = j.get('description','')
                    except:
                        desc = ''
            else:
                desc = ''
            #print(f"Found past checkpoint: {pc.split(',')[1]}  (trial desc: {desc})")
        choices = [ ( "NONE", "No Pre-training"), ] + [ (pc.split(',')[1], f"{pc.split(',')[0]}  ({os.path.dirname(os.path.dirname(pc.split(',')[1]))})") for pc in sorted(past_checkpoints)]
        #print(choices)
        dresult = dlg.menu(choices=choices, text='Found past checkpoints. Select one to load, or NONE to start fresh.', title='Load Checkpoint', width=90, height=20)
        if dresult[0] != dlg.OK:
            print('Exiting...')
            sys.exit(0)

        checkpoint_to_load = dresult[1] if dresult[0] != "NONE" else ''

    else:
        checkpoint_to_load = ''

    dirlist = sorted(glob("data/*"))
    def readme_text(d):
        readme_file = os.path.join(d, 'readme.txt')
        print(f"Looking for readme file: {readme_file}")
        if os.path.exists(readme_file):
            with open(readme_file, 'r') as f:
                return f.read()
        else:
            return '--'

    choices=[ (os.path.basename(x), readme_text(x)) for x in dirlist if os.path.isdir(os.path.join(x,'holo')) ]
    #print(choices)
    code, direct = dlg.menu(choices=choices,
                               text='Select data directory (must contain holo/ and gt/ subfolders):',
                               title='Select Data Directory',
                               width=90,
                               height=20)
    if code != dlg.OK:
        print('Exiting...')
        sys.exit(0)

    data_dir = os.path.join('data',direct)
    argdict['data_dir'] = data_dir

    if checkpoint_to_load != '':
        argdict['load_checkpoint'] = checkpoint_to_load

    #input()
    # label, yl, xl, item, yi, xi, field_length, input_length
    dialog_elements = [ (arglist[i], i+1,1, str(argdict[arglist[i]]), i+1,25,50,50) for i in range(len(arglist)) ]



    print("--- dialog_elements ---")
    print(dialog_elements)
    print("-----------------------")
    ret = dlg.form("training parameters", dialog_elements, title="Training Parameters") #, height=28, width=90)# form_height=24)
    if ret[0] != dlg.OK:
        print('Exiting...')
    print(ret[1])

    # check boolean args: convert 'True'/'False' strings to bool
    for i in range(len(arglist)):
        if argtypes[i] is bool:
            if ret[1][i].lower() == 'true':
                ret[1][i] = True
            else:
                ret[1][i] = False
        if argtypes[i] is int:
            ret[1][i] = int(ret[1][i])
        if argtypes[i] is float:
            ret[1][i] = float(ret[1][i])

    for i in range(len(arglist)):
        argdict[arglist[i]] = ret[1][i]

    print('Final training parameters:')
    for k in argdict:
        print(f'  {k}: {argdict[k]}')
    args = argparse.Namespace(**argdict)
    trial_json_dict = vars(args)
    # save trial config 
    trial_dir = os.path.dirname(args.checkpoints)
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, 'trial.json'), 'w') as f:
        json.dump(trial_json_dict, f, indent=4)
    with open(os.path.join(".", 'dihm_unet.json'), 'w') as f:
        json.dump(trial_json_dict, f, indent=4)

    cv2.namedWindow('validation',  cv2.WINDOW_NORMAL)
    cv2.resizeWindow('validation', 1200,400)
    cv2.moveWindow('validation', 0,0)
    device = torch.device(args.device)

    # dataset
    print(f"args.img_size = {args.img_size}")
    if not args.eval_only:
        t = make_transforms(args.img_size)
        ds = PairedHoloDataset(args.data_dir, transform=t)
        # simple random split 90/10
        n = len(ds)
        n_test = max(1, int(0.1 * n))
        n_val = max(1, int(0.1 * n))
        n_train = n - n_val - n_test
        train_ds, val_ds,test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test])
        if args.test_eval:
            test_loader = DataLoader  (test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        else:
            train_loader = DataLoader (train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = DataLoader   (val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        t = make_transforms(args.img_size)
        eval_ds = HoloDataset(args.data_dir, transform=None)
        eval_loader = DataLoader  (eval_ds,  batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = UNet(in_channels=3 if args.z_mm != 0 else 1, out_channels=1).to(device)
    model.apply(init_weights_to_random)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9)

    #scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    scaler = None
    print('Model summary:')
    print(f'input_size: (1, {args.in_ch}, {args.img_size}, {args.img_size})')
    print(f"model device = {next(model.parameters()).device}")
    model.to(device)

    print(f"MOD PARM: {next(model.parameters()).device}") # Check model device
    summary(model, input_size=(1, 3 if args.z_mm != 0 else 1, args.img_size, args.img_size), device=device)
    #sys.exit(0)
    best_val = float('inf')
    # model gradient reset
    model.zero_grad()
    # read checkpoint state if exists
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optim_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_val = checkpoint.get('best_val', float('inf'))
            print(f'Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}')
    print('Starting training...')
    print (args.test_eval)
    print(int(args.epochs)+1)
    print ("----------------")
    if args.eval_only:
        evaluate(model, eval_loader, device, 0)
        print('Evaluation complete.')
        return  
    for epoch in range(1, int(args.epochs)+1):
        model = model.float()
        model=model.to(device)
        model.zero_grad()
        if args.test_eval:
            test_loss = validate(model, test_loader, device, epoch)
            print(f'Test Loss: {test_loss:.6f}')
            break
        early_stop, train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler,epoch)
        val_loss = validate(model, val_loader, device, epoch)
        optimizer.step()
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        is_best = val_loss < best_val
        if is_best or early_stop:
            name = 'best' if is_best else 'earlystop'
            if is_best and early_stop:
                name = 'best_earlystop'
            best_val = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, args.checkpoints, name=f'epoch{epoch:03d}_{name}.pth')
        best_indicator = '****' if is_best else '    '
        print(f'Epoch {epoch:03d}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  Best: {best_val:.6f}{best_indicator} LR: {current_lr:.6e}')
        if not os.path.exists(os.path.join(trial_dir, 'training_log.csv')):
            print('epoch,timestamp,train_loss,val_loss,best_val,learning_rate,is_best', file=open(os.path.join(trial_dir, 'training_log.csv'), 'w'))
        nowstamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{epoch},{nowstamp},{train_loss:.6f},{val_loss:.6f},{best_val:.6f},{current_lr:.6e},{'1' if is_best else '0'}",
              file=open(os.path.join(trial_dir, 'training_log.csv'), 'a'))

    if not args.test_eval:
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'val_loss': val_loss,
        }, args.checkpoints, name=f'epoch{epoch:03d}_final.pth')
    print('Training complete. Best val loss:', best_val)


if __name__ == '__main__':

    main()
