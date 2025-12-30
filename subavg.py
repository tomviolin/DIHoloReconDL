#!/usr/bin/env python3
import sys

import cv2
import numpy as np
from glob2 import glob
import os,sys,shutil

from dialog import Dialog

dlg = Dialog(dialog="dialog")

def main():
    datadirs = glob("data/*/raw/")
    if not datadirs:
        print("No data directories found in 'data/*/raw'")
        sys.exit(1)

    dresult = dlg.menu("Select a dataset to process:", choices=[(d.replace('/raw/',''),)*2 for d in datadirs])
    if dresult[0] != dlg.DIALOG_OK:
        print("")
        sys.exit(1)
    data_dir = dresult[1]
    rawdir = os.path.join( data_dir, "raw")
    outdir = os.path.join( data_dir, "holo")
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
    for rawfile in rawfiles:
        print(f"Reading {rawfile}...")
        img = cv2.imread(rawfile, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {rawfile}, skipping.")
            continue
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

