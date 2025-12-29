#!/usr/bin/env python3

import cv2
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import glob
#import cupy as cp
import datetime
import argparse

def fixhololevel(pdata):
    data = None
    # grayscale only
    if len(pdata.shape) > 2:
        data = np.array((pdata[...,1]).astype(np.float32))
        # data = np.float32(cv2.cvtColor(pdata,cv2.COLOR_BGR2GRAY))
    else:
        data=pdata.copy()
    if type(data) is np.ndarray:
        data = np.array(data)
    data = np.abs(np.float32(data)) 
    # normalize to 0..1
    #data -= data.min()
    data /= data.max()
    # compute histogram
    datahst = np.histogram(data, bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(mostcommon)
    data = data ** n

    data = data - data.min()
    data = data / data.max()

    # compute histogram
    datahst = np.histogram(data, bins=256, range=(0.0, 1.0))[0]
    # find most common value
    mostcommon = np.argmax(datahst)/256
    # shift so that most common value is at 0.5

    n = np.log(0.5) / np.log(mostcommon)
    data = data ** n



    return data


def parse_args():
    p = argparse.ArgumentParser()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    p.add_argument("--now", type=str, default=now, help="current timestamp for data directory")
    p.add_argument("--objects_dir", type=str, required=True, help="directory containing object images to composite")
    p.add_argument('--data_dir', type=str, default=f"data/comp_{now}", help='output data dir containing holo/ and gt/')
    p.add_argument('--nimages', type=int, default=1000, help='number of composite images to generate')
    p.add_argument('--nobjects', type=int, default=50, help='number of objects to composite per image')
    p.add_argument('--object_size_range', type=int, default=32, help='scaling factor for object size')
    p.add_argument('--object_size_min', type=int, default=16, help='minimum object size')
    p.add_argument('--object_size_gamma', type=float, default=1.0, help='gamma correction for object size distribution')
    p.add_argument('--composite_size', type=int, default=256, help='size of composite image (square)')
    p.add_argument('--composite_padding', type=int, default=32, help='padding around composite image to ensure edge effects')
    return p.parse_args()


args=parse_args()

srcfiles = list(glob.glob(f"{args.objects_dir}/*.png"))

# shuffle source files
np.random.shuffle(srcfiles)
# make a selection of files
files = srcfiles.copy()
# make sure we have enough files
while len(files) < args.nobjects:
    files = files + files
# make sure our target directories exist
os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(f"{args.data_dir}/gt", exist_ok=True)
imgcache = {}
for numpics in range(1,args.nimages+1):
    imgout = np.zeros((args.composite_size+args.composite_padding*2,
                       args.composite_size+args.composite_padding*2) , dtype=np.float32)
    np.random.shuffle(files)
    thesefiles = files[:args.nobjects]
    np.random.shuffle(thesefiles)
    for file in thesefiles:
        #print(f"Processing {file}...")
        if file in imgcache:
            img = imgcache[file]
        else:
            imgcache[file] = fixhololevel(np.float32(cv2.imread(file, cv2.IMREAD_GRAYSCALE)/255.0))
            print(f"min: {imgcache[file].min()} max: {imgcache[file].max()}")
            print("m",end='',flush=True)
            img = imgcache[file]
        if img is None:
            print(f"Failed to load image {file}")
            continue

        # random flips
        if np.random.randint(2) == 1:
            img = cv2.flip(img, 0)
        if np.random.randint(2) == 1:
            img = cv2.flip(img, 1)
        if np.random.randint(2) == 1:
            img = img.T
        # random rotation
        angle = np.random.randint(0,360)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        sizescale = int(np.random.uniform(0.2,1.0)**args.object_size_gamma * args.object_size_range + args.object_size_min)
        imgtiny = (cv2.resize(fixhololevel(img), (sizescale, sizescale), interpolation=cv2.INTER_CUBIC))
        #imgtiny = (img*256).astype(np.uint8)
        # show original image
        #cv2.imshow("Original Image", imgtiny)
        #key = cv2.waitKey(1)
        #if key == 27:  # ESC key to exit
        #    break
        #imgtiny = img
        # place imgtiny into imgout at random position
        x = int(np.random.randint(0, imgout.shape[1] - imgtiny.shape[1]))
        y = int(np.random.randint(0, imgout.shape[0] - imgtiny.shape[0]))
        imgout[y:y+imgtiny.shape[0], x:x+imgtiny.shape[1]] += (imgtiny-0.5)
        #cv2.imshow("GDS Output", imgout)
    imgout += 0.5
    imgout = np.clip(imgout, 0.0, 1.0)
    imgout = (imgout*256).astype(np.uint8)
    outputdir = f"{args.data_dir}/gt"
    print(f"Saving {outputdir}/gds_output{numpics:05d}.png")
    cpad = args.composite_padding
    isz = args.composite_size
    cv2.imwrite(f"{outputdir}/gds_output{numpics:05d}.png", imgout[cpad:isz-cpad,cpad:isz-cpad])
