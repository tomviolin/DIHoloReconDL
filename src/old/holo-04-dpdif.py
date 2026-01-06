#!/usr/bin/env python3

import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cupy
import context  # Loads relative paths for pyholoscope
import pyholoscope as pyh

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

img_size = 512
fftpad = 0
fftp = 8
#holo = np.float32(cv2.imread("composites/holo/gds_output00006.png")/255)[...,0]
#kernel = np.float32(cv2.imread("onepix/holo/onepixel01.png")/255.0)[...,0]
#gt = np.float32(cv2.imread("composites/gt/gds_output00006.png")/255.0)[...,0]

# establish values that will not change based on holo params
xx = np.linspace(-img_size//2,img_size//2-1,img_size)
yy = np.linspace(-img_size//2,img_size//2-1,img_size)
x,y = np.meshgrid(xx,yy)
padding = np.ones([img_size+fftpad*2]*2)*0.5

hologram = pyh.load_image(sys.argv[1])
hologram = fixhololevel(hologram)
#holo = fixhololevel(np.float32(cv2.imread(sys.argv[1])/255)[...,0])
datahome, basename = os.path.split(sys.argv[1])
datahome, datalastdir = os.path.split(datahome)



# holo params
wavelen = 635.0e-9
dx = 1.12e-6
r = np.sqrt(x*x + y*y) * dx


if fftpad > 0:
    paddedholo = padding.copy()
    paddedholo[fftpad:-fftpad,fftpad:-fftpad] = hologram
else:
    paddedholo = hologram.copy()

#zees = np.arange(1e-3-wavelen*1600,6e-3+wavelen*1000,wavelen*500)

zees = np.arange(0.0002, 0.003, 0.0002) 
#zees = np.arange(wavelen*100, wavelen*5000, wavelen*100)
#zees = np.arange(0.001, 0.003, 0.0001) 
#zees = np.float32([0.0028])
#zees = np.float32((0.00328, 0.00328001))

def makefig(zi):
    # these will change
    z = zees[zi]
    d = np.sqrt(r*r+z*z)

    # Create an instance of the Holo class
    holo = pyh.Holo(
        mode=pyh.INLINE,  # For inline holography
        wavelength=wavelen,  # Light wavelength, m
        pixel_size=dx,  # Hologram physical pixel size, m
    #    background=background,  # To subtract the background
        depth=zees[zi],
        cuda=True
    )  # Distance to refocus, m

    # Refocus
    recon = holo.process(hologram)


    #fig,ax = plt.subplots(1,2)
    #fig.set_size_inches(12,6)
    cimage=recon #np.stack((im/2,im/2+edgesim/2,im/2+edgesim/2),axis=2)
    #ax[0].imshow(gt,cmap='binary_r')
    #ax[0].set_title('gt',fontsize=30)
    #ax[1].imshow(holo,cmap='binary_r')
    #ax[1].oset_title('holo',fontsize=30)
    print (f"\n****** [{zi}/{len(zees)}] z={z:8.60f} ******")
    print(f"mkdir {datahome}/cimage")
    print(f"save to {datahome}/cimage/z{zi:05d}_{basename}")
    os.makedirs(f"{datahome}/cimage", exist_ok=True)
    cimadj = np.uint8(np.abs(cimage)*255)
    cv2.putText(cimadj,f"z={z:09.07f} zi={zi:05f} frame={sys.argv[1]}", (1,51),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0),2)
    cv2.putText(cimadj,f"z={z:09.07f} zi={zi:05f} frame={sys.argv[1]}", (0,50),cv2.FONT_HERSHEY_DUPLEX,2,(0,1,1),2)
    cv2.imwrite(f"{datahome}/cimage/{basename}",cimadj)
    print("******************")
    return(True)

for zi in range(len(zees)):
    makefig(zi)
