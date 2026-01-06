#!/usr/bin/env python3
import sys

import cv2
import numpy as np
from glob2 import glob
import os,sys,shutil

from dialog import Dialog

dlg = Dialog(dialog="whiptail")


def fixmedian(img):
    img=img-img.min()
    img=img/img.max()
    for i in range(1):
        median = np.median(img)
        factor = np.log(0.5)/np.log(median)
        fixed = img ** factor
        img= fixed
    return img

def main():
    datadirs = glob("data/*/raw/")
    if not datadirs:
        print("No data directories found in 'data/*/raw'")
        sys.exit(1)
    choices = []
    for d in datadirs:
        dh = d.split('/')[1]
        raws = glob(os.path.join(d, "*"))
        if not raws:
            continue
        rc = len(raws)
        desc = "[{} raw]".format(rc)
        holos = glob( os.path.join( d.replace("/raw/","/holo/"), "*") )
        hc = len(holos)
        if hc > 0:
            desc = desc + " [{} holo]".format(hc)
        else:
            desc = desc + " \\Zb\\Z1[     ]\\Zn".format(hc)
        if os.path.exists( os.path.join(d, "readme.txt") ):
            with open( os.path.join(d, "readme.txt"), 'r') as f:
                firstline = f.readline().strip()
                desc = desc + " " +firstline
        choices.append( (dh, desc) )
    dresult = dlg.menu("Select a dataset to process:", choices=choices)
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

    sum_image = None
    count = 0
    imgs = []
    imgwindow = []
    windowradius = 50
    windowsize = windowradius*2 + 1
    lastimg = None
    for rawfile in rawfiles:
        print(f"Reading {rawfile}...")
        img = cv2.imread(rawfile, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        img = fixmedian(img)
        if img is None:
            print(f"Failed to read {rawfile}, skipping.")
            continue
        imgs.append(fixmedian(img.copy()))

    imgavg = np.sum(imgs, axis=0) / len(imgs)
    imgavg = fixmedian(imgavg)
    cv2.imshow("Image Average Window", (imgavg*255).astype(np.uint8))
    if cv2.waitKey(5) == 27:
        print("Process interrupted by user.")
        sys.exit(0)

    for idx, img in enumerate(imgs):
        img = (img+0.1) / (imgavg+0.1)
        img = fixmedian(img)

        filename = os.path.basename(rawfiles[idx])
        outpath = os.path.join(outdir, filename)
        cv2.imwrite(outpath.replace(".jpg",".png"), np.uint8(img*255))
    sys.exit(0)
    if False:
        img = img.astype(np.float32)/255.0
        if sum_image is None:
            sum_image = np.zeros_like(img)
        sum_image += img
        count += 1

    if count == 0:
        print("No valid images were processed.")
        sys.exit(1)

    avg_image = sum_image / count
    
    for rawfile in rawfiles:
        filename = os.path.basename(rawfile)
        outpath = os.path.join(outdir, filename)
        img = cv2.imread(rawfile, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

        subtracted = (img+1) / (avg_image+1)
        subtracted -= subtracted.min()
        subtracted /= subtracted.max()
        
        print(f"pixel range: min {subtracted.min()} max {subtracted.max()}")
        subtracted = np.clip(subtracted, 0, 1)
        subtracted = (subtracted * 255).astype(np.uint8)
        cv2.imwrite(outpath.replace(".jpg",".png"), subtracted)
        print(f"Saved processed image to {outpath}")



if __name__ == "__main__":
    main()

