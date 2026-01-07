#!/usr/bin/env python3
import sys

import cv2
import numpy as np
from glob2 import glob
import os,sys,shutil

from dialog import Dialog

dlg = Dialog(dialog="dialog")


def fixhololevel(img):
    img=img-img.min()
    img=img/img.max()
    h = np.histogram(img, bins=256, range=(0,1))
    modebin = np.argmax(h[0])
    mode = (h[1][modebin] + h[1][modebin+1]) / 2
    exponent = np.log(0.5)/np.log(mode)
    img = img ** exponent
    return img

def main():
    datadirs = glob("data/*/raw/")
    if not datadirs:
        print("No data directories found in 'data/*/raw'")
        sys.exit(1)
    choices = []
    for d in datadirs:
        dh = d.split('/')[1]
        raws = sorted(glob(os.path.join(d, "*")))
        if not raws:
            continue
        rc = len(raws)
        desc = "[{:4d} raw]".format(rc)
        holos = glob( os.path.join( d.replace("/raw/","/holo/"), "*") )
        hc = len(holos)
        if hc > 0:
            desc = desc + " [{:4d} holo]".format(hc)
        else:
            desc = desc + " [         ]".format(hc)
        if os.path.exists( os.path.join(d, "readme.txt") ):
            with open( os.path.join(d, "readme.txt"), 'r') as f:
                firstline = f.readline().strip()
                desc = desc + " " +firstline
        choices.append( (dh, desc) )
    choices = sorted(choices)
    dresult = dlg.menu("Select a dataset to process:", choices=choices, width=78)
    if dresult[0] != dlg.DIALOG_OK:
        print("")
        sys.exit(1)
    data_dir = dresult[1]
    rawdir = os.path.join( "data", data_dir, "raw")
    outdir = os.path.join( "data", data_dir , "holo")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f"Processing dataset: {data_dir}")
    print(f"Raw directory: {rawdir}")
    print(f"Output directory: {outdir}")

    rawfiles = sorted(glob(os.path.join(rawdir, "*.jpg"))
                    + glob(os.path.join(rawdir, "*.png")))
    
    if not rawfiles:
        print(f"No image files found in {rawdir}")
        sys.exit(1)

    imgs = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for f in rawfiles])

    imgs = fixhololevel(imgs)

    imgs = np.clip(imgs, 0, 1)
    imgs = imgs - imgs.mean(axis=0)
    imgs = imgs / np.max(np.abs(imgs))*0.5 + 0.5

    print(f"value range: min {imgs.min()} max {imgs.max()}")
    for i in range(imgs.shape[0]):
        filename = os.path.basename(rawfiles[i])
        outpath = os.path.join(outdir, filename)
        cv2.imwrite(outpath.replace(".jpg",".png"), np.uint8(imgs[i]*255))
        print(f"Saved processed image to {outpath}")


if __name__ == "__main__":
    main()

