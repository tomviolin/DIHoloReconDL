#!/usr/bin/env python3

import os,sys
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import cupy
import cupy as cp

def fixhololevel(pdata):
    data = None
    # grayscale only
    if len(pdata.shape) > 2:
        data = cp.array((pdata[...,1]).astype(cp.float32))
        # data = cp.float32(cv2.cvtColor(pdata,cv2.COLOR_BGR2GRAY))
    else:
        data=pdata.copy()
    if type(data) is np.ndarray:
        data = cp.array(data)
    
    print(type(data))
    # normalize to 0..1
    data -= data.min()
    data /= data.max()
    # compute histogram
    print(type(data))
    datahst = cp.histogram(data, bins=256, range=(0.0, 1.0))[0]
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
pimgsize = img_size + fftpad*2
xx = cp.linspace(-pimgsize//2,pimgsize//2-1,pimgsize)
yy = cp.linspace(-pimgsize//2,pimgsize//2-1,pimgsize)
x,y = cp.meshgrid(xx,yy)
padding = cp.ones([img_size+fftpad*2]*2)*0.5
padding = padding + 0j


# holo params
wavelen = 650.0e-9
dx = 1.12e-6
r = cp.sqrt(x*x + y*y) * dx

zees = np.arange(wavelen*10, 0.01, wavelen) 
zi=0

def makefig(paddedholo, zi, imgptr):
    # these will change
    global zees, r
    z = zees[zi]
    d = cp.sqrt(r*r+z*z)

    print (f"\n****** [{zi}/{len(zees)}] z={z:8.60f} ******")


    # self-computed kernel
    kmask= 1.0 / (d+1)
    amp = -cp.sin((d / wavelen) * cp.pi*2)  * kmask
    amp = amp / 2.0 + 0.5

    phase = d * np.pi * 2 / wavelen

    amp = amp*cp.cos(phase) + 1j * amp * cp.sin(phase)
    
    #if fftpad > 0:
    #    paddedamp =padding.copy()
        #paddedamp[fftpad:-fftpad,fftpad:-fftpad] = amp
    #else:
    paddedamp = amp.copy()

    paddedLaplacian = padding.get().copy()
    pdh = padding.shape[0]//2
    #pdh = int(pdh)
    paddedLaplacian[...] = 0
    lap = cp.float32(
               [[ 0, 1, 0 ],
                [ 1, -4, 1 ],
                [ 0, 1, 0]]
          )
    paddedLaplacian[pdh-1:pdh+2, pdh-1:pdh+2] = lap


    # Kernel FFT
    kernelFFT = cupy.fft.fft2(cupy.array(paddedamp))
    print(f"kernelFFT.shape={kernelFFT.shape} kernelFFT.dtype={kernelFFT.dtype}")
    print(f"np.real(kernelFFT).shape={np.real(kernelFFT).shape} np.real(kernelFFT).dtype={np.real(kernelFFT).dtype}")

    # FFT of hologram
    holoFFT = cupy.fft.fft2(cupy.array(paddedholo))

    # FFT of Laplacian
    LapFFT = cupy.fft.fft2(cupy.array(paddedLaplacian))

    # Product of KernelFFT and holoFFT => 
    #    FFT of the convolution of the kernel and the hologram
    prodFFT = kernelFFT * holoFFT

    # FFT of the convolution of the Laplacian over the hologram reconstruction
    #edgesFFT = prodFFT * LapFFT * LapFFT 

    # inverse FFT of the product of the Kernel and the hologram 
    #   is the kernel convolved over the hologram
    prodpadded = cupy.fft.fftshift(cupy.fft.ifft2(prodFFT))

    # the inverse FFT of the product of the Ker
    #edgesPadded = cupy.fft.fftshift(cupy.fft.ifft2(edgesFFT))

    prod=prodpadded[fftp:-fftp,fftp:-fftp]
    #edges = edgesPadded[fftp:-fftp,fftp:-fftp]
    #print(prod[:5,:5])

    im = cupy.abs(prod)
    im = fixhololevel(im)

    #edgesim = (np.abs(edges))
    #edgesim = cv2.absdiff(edgesim.get(),np.mean(edgesim.get()))

    #edgesblur = cv2.GaussianBlur(im, (0,0), sigmaX=3, sigmaY=3)
    #edgesblur2 =cv2.GaussianBlur(im, (0,0), sigmaX=5, sigmaY=5)
    #edgesdiff = cv2.absdiff(edgesblur,edgesblur2)
    #edgesim = edgesdiff
    #edgesim[edgesim<np.quantile(edgesim,.99)]=0
    #alpha=0.8
    #edgesim =  (edgesim * (1.5+alpha) + ( edgesblur * (-0.5) ))
    cimage = im# * edgesim
    cdiff = fixhololevel(cimage)
            #cp.abs(cimage-0.5)
    #edgesim=cdiff


    #fig,ax = plt.subplots(1,2)
    #fig.set_size_inches(12,6)
    #cimage=recon #np.stack((im/2,im/2+edgesim/2,im/2+edgesim/2),axis=2)
    #ax[0].imshow(gt,cmap='binary_r')
    #ax[0].set_title('gt',fontsize=30)
    #ax[1].imshow(holo,cmap='binary_r')
    #ax[1].oset_title('holo',fontsize=30)
    print("******************")
    print(f"mkdir {datahome}/cimage")
    os.makedirs(f"{datahome}/cimage", exist_ok=True)
    cimadj = np.uint8(cp.abs(cimage).get()*255.0)
    finame = f"{datahome}/cimage/z{zi:04d}_{basename}"
    print(f"save to {finame}")
    """
    .   @param img Image.
    .   @param text Text string to be drawn.
    .   @param org Bottom-left corner of the text string in the image.
    .   @param fontFace Font type, see #HersheyFonts.
    .   @param fontScale Font scale factor that is multiplied by the font-specific base size.
    .   @param color Text color.
    .   @param thickness Thickness of the lines used to draw a text.
    .   @param lineType Line type. See #LineTypes
    .   @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,
    .   it is at the top-left corner.
    """
    #cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:05f} frame={sys.argv[imgptr]}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,0),2)
    #cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:05f} frame={sys.argv[imgptr]}", (0,50),cv2.FONT_HERSHEY_DUPLEX,0.4,(255,255,255),2)
    cv2.imwrite(f"{finame}",cimadj)
    #os.makedirs(f"{datahome}/edgesim", exist_ok=True)
    #cv2.imwrite(f"{datahome}/edgesim/{basename}",cimadj)
    print("******************")
    return cimage


### per image execution here ###
imgptr = 1
escaped = False
while not escaped:
    if len(sys.argv) < 2:
        break
    hologram = cp.array(cv2.imread(sys.argv[imgptr]))

    hologram = fixhololevel(hologram)
    datahome, basename = os.path.split(sys.argv[imgptr])
    datahome, datalastdir = os.path.split(datahome)

    if fftpad > 0:
        paddedholo = padding.copy()
        paddedholo[fftpad:-fftpad,fftpad:-fftpad] = hologram
    else:
        paddedholo = hologram.copy()

    cimage = makefig(paddedholo, zi, imgptr)
    if zi > 0:
        cimbelow = makefig(paddedholo, zi-1, imgptr)
    else:
        cimbelow = cimage
    if zi < len(zees)-1 :
        cimabove = makefig(paddedholo, zi+1, imgptr)
    else:
        cimabove = cimage

    avgfile = (cimabove + cimbelow) / 2.0
    thisfile = fixhololevel((cimage+0.01) / (avgfile+0.01))

    cimadj = thisfile.get()
    cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:05f} frame={sys.argv[imgptr]}", (1,51),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,0),2)
    cv2.putText(cimadj,f"z={zees[zi]:09.06f} zi={zi:05f} frame={sys.argv[imgptr]}", (0,50),cv2.FONT_HERSHEY_DUPLEX,0.4,(255,255,155),1)

    while True:
        if type(cimadj) is not np.ndarray:
            cimadj = cimadj.get()
        cv2.imshow("image", cimadj)
        k=cv2.waitKey(0) & 255
        if k == 27 or k == ord('q'):
            escaped=True
            break
        if k == ord('h'):
            if imgptr > 1:
                imgptr -= 1
                break
        if k == ord('l'):
            if imgptr < len(sys.argv) - 1:
                imgptr += 1
                break
        if k == ord('j'):
            if zi > 0:
                zi -= 1
                break
        if k == ord('k'):
            if zi < len(zees) - 1:
                zi += 1
                break
        if k == ord('n'):
            if zi > 10:
                zi -= 10
            else:
                zi  = 0
            break
        if k == ord('i'):
            if zi < len(zees) - 11:
                zi += 10
            else:
                zi = len(zees)-1
            break

