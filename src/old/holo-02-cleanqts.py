#!/usr/bin/env python3

import cv2
import numpy as np
from glob import glob

import os,sys

thisdir, thisfile = os.path.split(sys.argv[0])

os.chdir(os.path.realpath(thisdir + "/.."))
print(f"dir = {os.getcwd()}")
datahome = sys.argv[1]
print(f"datahome = {datahome}")

#timestamp= now.strftime("%Y-%m-%d-%H:%M:%S")


def fixhololevel(pdata):
    # grayscale only
    if len(pdata.shape) > 2:
        data = np.float32(cv2.cvtColor(data,cv2.COLOR_BGR2GRAY))
    else:
        data=np.float32(pdata.copy())
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


files=sorted(glob(f"{datahome}/cropped/*.png"))
print(f"files={files}")
avgfile=None
avgcount = 0
for f in files:
    thisfile = fixhololevel(cv2.imread(f,cv2.IMREAD_GRAYSCALE))
    print (f"read {f}!")
    if avgfile is None:
        avgfile = thisfile
        avgcount  = 1
    else:
        avgfile = avgfile + thisfile
        avgcount += 1

avgfile = fixhololevel(avgfile)

lastfile = None
lastsumdiff = None
for f in files:
    thisfile = fixhololevel(cv2.imread(f,cv2.IMREAD_GRAYSCALE))

    thisfile = fixhololevel((thisfile+0.01) / (avgfile+0.01))

    if lastfile is not None:
        print(f"lastfile.shape={lastfile.shape} lastfile.dtype={lastfile.dtype}")
        print(f"thisfile.shape={thisfile.shape} thisfile.dtype={thisfile.dtype}")
        sumdiff = np.sum(cv2.absdiff(np.float32(lastfile),np.float32(thisfile)).flatten()) 
    else:
        sumdiff = 0

    print(sumdiff,end=' ',flush=True)
    print(f" **L={lastsumdiff} ",end='',flush=True)
    os.makedirs(f"{datahome}/holo/",exist_ok=True)
    if True or lastsumdiff is None or sumdiff > lastsumdiff:
        savef = f.split('/')
        savefile = f"{datahome}/holo/"+savef[-1] # +".png"
        print(f"SAVING to {savefile}")
        cv2.imwrite(savefile, np.uint8(thisfile*255))
        print(f"SAVED to {savefile}")
    print('') 
    lastsumdiff = sumdiff
    cv2.imshow('l',thisfile)
    if lastfile is None:
        lastfile = thisfile
    else:
        lastfile = lastfile *0.95 + thisfile*0.05
    k=cv2.waitKey(0)
    if k == 27:
        break

cv2.imshow('l',avgfile)
k=cv2.waitKey(0)













































